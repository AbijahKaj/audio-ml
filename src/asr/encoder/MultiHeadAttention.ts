import type { ComputeBackend, TensorHandle } from '../compute/index.js';
import { ComputeScope } from '../compute/index.js';
import type { AttentionWeights } from '../model/WeightMapper.js';
import { Linear } from './Linear.js';
import { RelativePositionalEncoding } from './RelativePositionalEncoding.js';

/**
 * Multi-Head Self-Attention with Relative Positional Encoding.
 *
 * Implements NeMo's RelPositionMultiHeadAttention:
 *
 *   score[i,j] = scale * ( (q_i + u)·k_j^T + (q_i + v)·pe[i-j]^T )
 *              = AC term + BD term
 *
 * Input:  x [B, T, d_model]
 * Output: [B, T, d_model]
 *
 * For streaming mode, caller passes cachedK/cachedV which are prepended to K/V.
 */
export class MultiHeadAttention {
  private readonly numHeads: number;
  private readonly headDim: number;
  private readonly dModel: number;
  private readonly scale: number;

  private readonly norm: { weight: TensorHandle; bias: TensorHandle };
  private readonly qProj: Linear;
  private readonly kProj: Linear;
  private readonly vProj: Linear;
  private readonly outProj: Linear;
  private readonly posProj: Linear;
  private readonly posU: TensorHandle;
  private readonly posV: TensorHandle;
  private readonly relPosEnc: RelativePositionalEncoding;

  constructor(
    private readonly backend: ComputeBackend,
    weights: AttentionWeights,
    dModel: number,
    numHeads: number,
  ) {
    this.dModel = dModel;
    this.numHeads = numHeads;
    this.headDim = dModel / numHeads;
    this.scale = 1.0 / Math.sqrt(this.headDim);

    this.norm = weights.norm;
    this.qProj = Linear.fromWeights(backend, weights.queryProj);
    this.kProj = Linear.fromWeights(backend, weights.keyProj);
    this.vProj = Linear.fromWeights(backend, weights.valueProj);
    this.outProj = Linear.fromWeights(backend, weights.outProj);
    this.posProj = Linear.fromWeights(backend, weights.posProj);
    this.posU = weights.posU;
    this.posV = weights.posV;
    this.relPosEnc = new RelativePositionalEncoding(backend, dModel);
  }

  forward(
    x: TensorHandle,
    cachedK?: TensorHandle,
    cachedV?: TensorHandle,
    attentionMask?: TensorHandle,
  ): {
    output: TensorHandle;
    newK: TensorHandle;
    newV: TensorHandle;
  } {
    const scope = new ComputeScope();
    const shape = this.backend.getShape(x);
    const B = shape[0] as number;
    const T = shape[1] as number;

    // Layer norm
    const normed = scope.track(
      this.backend.layerNorm(x, this.norm.weight, this.norm.bias, 1e-5),
    );

    // Project Q, K, V
    const q = scope.track(this.qProj.forward(normed));
    const k = scope.track(this.kProj.forward(normed));
    const v = scope.track(this.vProj.forward(normed));

    // Concatenate cached K, V if streaming
    let fullK: TensorHandle;
    let fullV: TensorHandle;
    if (cachedK && cachedV) {
      fullK = scope.track(this.backend.concat([cachedK, k], 1));
      fullV = scope.track(this.backend.concat([cachedV, v], 1));
    } else {
      fullK = k;
      fullV = v;
    }
    const Tkv = this.backend.getShape(fullK)[1] as number;

    // Split into heads: [B, T, dModel] → [B, H, T, d_head]
    const qH = scope.track(this.splitHeads(q, B, T));
    const kH = scope.track(this.splitHeads(fullK, B, Tkv));
    const vH = scope.track(this.splitHeads(fullV, B, Tkv));

    // Build attention scores (AC + BD)
    const scores = scope.track(this.computeAttentionScores(qH, kH, B, T, Tkv));

    // Apply causal/streaming mask if provided
    let maskedScores: TensorHandle;
    if (attentionMask) {
      maskedScores = scope.track(
        this.backend.add(scores, scope.track(this.backend.scale(attentionMask, -1e9))),
      );
    } else {
      maskedScores = scores;
    }

    const attnWeights = scope.track(this.backend.softmax(maskedScores, -1));
    const attnOut = scope.track(this.backend.matmul(attnWeights, vH));

    const merged = scope.track(this.mergeHeads(attnOut, B, T));
    const out = this.outProj.forward(merged);

    // Detach K/V from scope so they can be returned as cache
    scope.keep(fullK);
    scope.keep(fullV);

    scope.dispose(this.backend);
    return { output: out, newK: fullK, newV: fullV };
  }

  private computeAttentionScores(
    q: TensorHandle,
    k: TensorHandle,
    B: number,
    T: number,
    Tkv: number,
  ): TensorHandle {
    const scope = new ComputeScope();

    // AC term: (Q + u) · K^T  scaled
    const uExp = scope.track(this.backend.reshape(this.posU, [1, this.numHeads, 1, this.headDim]));
    const qu = scope.track(this.backend.add(q, uExp));
    const kT = scope.track(this.backend.transpose(k, [0, 1, 3, 2]));
    const ac = scope.track(this.backend.matmul(qu, kT));
    const acScaled = scope.track(this.backend.scale(ac, this.scale));

    // BD term: (Q + v) · PE^T  scaled
    const maxLen = Math.max(T, Tkv);
    const posEnc = scope.track(this.relPosEnc.buildPositionEncoding(maxLen));

    const posEncFlat = scope.track(this.backend.reshape(posEnc, [1, 2 * maxLen - 1, this.dModel]));
    const posProj2 = scope.track(this.posProj.forward(posEncFlat));
    const posProjReshaped = scope.track(
      this.backend.reshape(posProj2, [1, 2 * maxLen - 1, this.numHeads, this.headDim]),
    );
    const posProjT = scope.track(
      this.backend.transpose(posProjReshaped, [0, 2, 3, 1]),
    );  // [1, H, d_head, 2L-1]

    const vExp = scope.track(this.backend.reshape(this.posV, [1, this.numHeads, 1, this.headDim]));
    const qv = scope.track(this.backend.add(q, vExp));
    const bdRaw = scope.track(this.backend.matmul(qv, posProjT));  // [B, H, T, 2L-1]

    const bd = scope.track(this.relShift(bdRaw, B, this.numHeads, T, Tkv, maxLen));
    const bdScaled = scope.track(this.backend.scale(bd, this.scale));

    const scores = this.backend.add(acScaled, bdScaled);
    scope.dispose(this.backend);
    return scores;
  }

  private relShift(
    bd: TensorHandle,
    B: number,
    H: number,
    T: number,
    Tkv: number,
    maxLen: number,
  ): TensorHandle {
    const scope = new ComputeScope();
    const padded = scope.track(
      this.backend.pad(bd, [[0, 0], [0, 0], [0, 0], [1, 0]]),
    );
    const reshaped = scope.track(
      this.backend.reshape(padded, [B, H, 2 * maxLen, T]),
    );
    const sliced = scope.track(
      this.backend.slice(reshaped, [0, 0, 1, 0], [B, H, T, T]),
    );
    const out = Tkv === T
      ? this.backend.reshape(sliced, [B, H, T, T])
      : this.backend.slice(sliced, [0, 0, 0, 0], [B, H, T, Math.min(T, Tkv)]);
    scope.dispose(this.backend);
    return out;
  }

  private splitHeads(x: TensorHandle, B: number, T: number): TensorHandle {
    const scope = new ComputeScope();
    const r = scope.track(
      this.backend.reshape(x, [B, T, this.numHeads, this.headDim]),
    );
    const out = this.backend.transpose(r, [0, 2, 1, 3]);
    scope.dispose(this.backend);
    return out;
  }

  private mergeHeads(x: TensorHandle, B: number, T: number): TensorHandle {
    const scope = new ComputeScope();
    const t = scope.track(this.backend.transpose(x, [0, 2, 1, 3]));
    const out = this.backend.reshape(t, [B, T, this.dModel]);
    scope.dispose(this.backend);
    return out;
  }
}
