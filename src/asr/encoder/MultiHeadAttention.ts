import type { ComputeBackend } from '../compute/Backend';
import type { TensorHandle } from '../compute/types';
import type { RelPosAttnWeights } from '../model/types';
import { linearForward } from './linear';

/**
 * NeMo RelPositionMultiHeadAttention (non-SDPA path).
 */
export class RelPositionMultiHeadAttention {
  private readonly dModel: number;
  private readonly numHeads: number;
  private readonly headDim: number;
  private readonly scale: number;

  constructor(
    private readonly backend: ComputeBackend,
    private readonly weights: RelPosAttnWeights,
    dModel: number,
    numHeads: number,
  ) {
    this.dModel = dModel;
    this.numHeads = numHeads;
    this.headDim = dModel / numHeads;
    this.scale = 1 / Math.sqrt(this.headDim);
  }

  forward(x: TensorHandle, posEmb: TensorHandle): TensorHandle {
    const q = linearForward(this.backend, x, this.weights.linearQ);
    const k = linearForward(this.backend, x, this.weights.linearK);
    const v = linearForward(this.backend, x, this.weights.linearV);

    const qh = this.splitHeads(q);
    const kh = this.splitHeads(k);
    const vh = this.splitHeads(v);
    this.backend.dispose(q);
    this.backend.dispose(k);
    this.backend.dispose(v);

    const posProj = linearForward(this.backend, posEmb, this.weights.linearPos);
    const ph = this.splitHeadsPos(posProj);
    this.backend.dispose(posProj);

    const bu = this.backend.reshape(this.weights.posBiasU, [1, this.numHeads, 1, this.headDim]);
    const bv = this.backend.reshape(this.weights.posBiasV, [1, this.numHeads, 1, this.headDim]);

    const qU = this.backend.add(qh, bu);
    const qV = this.backend.add(qh, bv);
    this.backend.dispose(bu);
    this.backend.dispose(bv);

    const kt = this.backend.transpose(kh, [0, 1, 3, 2]);
    const matrixAc = this.backend.matmul(qU, kt);
    this.backend.dispose(kt);

    const pt = this.backend.transpose(ph, [0, 1, 3, 2]);
    const matrixBdRaw = this.backend.matmul(qV, pt);
    this.backend.dispose(pt);
    this.backend.dispose(qV);

    const matrixBd = this.relShift(matrixBdRaw);
    this.backend.dispose(matrixBdRaw);

    const acShape = this.backend.getShape(matrixAc);
    const bdTrim = this.backend.slice(matrixBd, [0, 0, 0, 0], [
      acShape[0]!,
      acShape[1]!,
      acShape[2]!,
      acShape[3]!,
    ]);
    this.backend.dispose(matrixBd);

    const scores = this.backend.scale(this.backend.add(matrixAc, bdTrim), this.scale);
    this.backend.dispose(matrixAc);
    this.backend.dispose(bdTrim);

    const attn = this.backend.softmax(scores, -1);
    this.backend.dispose(scores);

    const ctx = this.backend.matmul(attn, vh);
    this.backend.dispose(attn);
    this.backend.dispose(qU);
    this.backend.dispose(qh);
    this.backend.dispose(kh);
    this.backend.dispose(vh);
    this.backend.dispose(ph);

    const merged = this.mergeHeads(ctx);
    this.backend.dispose(ctx);

    const out = linearForward(this.backend, merged, this.weights.linearOut);
    this.backend.dispose(merged);
    return out;
  }

  private splitHeads(x: TensorHandle): TensorHandle {
    const s = this.backend.getShape(x);
    const batch = s[0]!;
    const t = s[1]!;
    const resh = this.backend.reshape(x, [batch, t, this.numHeads, this.headDim]);
    const tr = this.backend.transpose(resh, [0, 2, 1, 3]);
    this.backend.dispose(resh);
    return tr;
  }

  private splitHeadsPos(pos: TensorHandle): TensorHandle {
    const s = this.backend.getShape(pos);
    const batch = s[0]!;
    const tpos = s[1]!;
    const resh = this.backend.reshape(pos, [batch, tpos, this.numHeads, this.headDim]);
    const tr = this.backend.transpose(resh, [0, 2, 1, 3]);
    this.backend.dispose(resh);
    return tr;
  }

  private mergeHeads(x: TensorHandle): TensorHandle {
    const s = this.backend.getShape(x);
    const batch = s[0]!;
    const t = s[2]!;
    const tr = this.backend.transpose(x, [0, 2, 1, 3]);
    const merged = this.backend.reshape(tr, [batch, t, this.dModel]);
    this.backend.dispose(tr);
    return merged;
  }

  /**
   * NeMo rel_shift: x is [B, H, qlen, 2*qlen-1]
   */
  private relShift(x: TensorHandle): TensorHandle {
    const s = this.backend.getShape(x);
    const b = s[0]!;
    const h = s[1]!;
    const qlen = s[2]!;
    const posLen = s[3]!;

    const padded = this.backend.pad(x, [
      [0, 0],
      [0, 0],
      [0, 0],
      [1, 0],
    ]);
    const swapped = this.backend.transpose(padded, [0, 1, 3, 2]);
    this.backend.dispose(padded);

    const sliced = this.backend.slice(swapped, [0, 0, 1, 0], [b, h, posLen, qlen]);
    this.backend.dispose(swapped);

    const out = this.backend.transpose(sliced, [0, 1, 3, 2]);
    this.backend.dispose(sliced);
    return out;
  }
}