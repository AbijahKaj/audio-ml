import type { ComputeBackend, TensorHandle } from '../compute/index.js';
import { ComputeScope } from '../compute/index.js';

/**
 * Relative Positional Encoding for FastConformer.
 *
 * Based on the XL-style relative position bias used in NeMo's RelPositionMultiHeadAttention.
 * Reference: Pham et al. (Interspeech 2020) — "Relative Positional Encoding for Speech
 * Recognition and Direct Translation".
 *
 * This module computes the sinusoidal position encoding table PE[pos, d] and is used
 * inside MultiHeadAttention to compute the AC and BD terms:
 *
 *   score = (Q + u) · K^T + (Q + v) · PE^T
 *
 * where u, v are learnable per-head biases.
 */
export class RelativePositionalEncoding {
  private readonly dModel: number;

  constructor(
    private readonly backend: ComputeBackend,
    dModel: number,
  ) {
    this.dModel = dModel;
  }

  /**
   * Build sinusoidal position encoding for positions 0..maxLen-1.
   *
   * Returns tensor of shape [1, 2*maxLen-1, dModel] — the "full" relative
   * position matrix covering negative and positive offsets.
   */
  buildPositionEncoding(maxLen: number): TensorHandle {
    const scope = new ComputeScope();
    const dModel = this.dModel;

    // Build for positions -(maxLen-1) to +(maxLen-1)
    const numPos = 2 * maxLen - 1;
    const pe = new Float32Array(numPos * dModel);

    for (let pi = 0; pi < numPos; pi++) {
      const pos = pi - (maxLen - 1);  // centred: -(maxLen-1) .. 0 .. +(maxLen-1)
      for (let i = 0; i < dModel / 2; i++) {
        const omega = Math.pow(10000, (-2 * i) / dModel);
        pe[pi * dModel + 2 * i] = Math.sin(pos * omega);
        pe[pi * dModel + 2 * i + 1] = Math.cos(pos * omega);
      }
    }

    const peTensor = scope.track(this.backend.tensor(pe, [numPos, dModel]));
    const out = this.backend.reshape(peTensor, [1, numPos, dModel]);
    scope.dispose(this.backend);
    return out;
  }

  /**
   * Compute relative position attention scores.
   *
   * @param q        Query tensor [B, H, T, d_head]
   * @param posEnc   Position encoding [1, 2T-1, dModel] from buildPositionEncoding()
   * @param posProj  Position projection [dModel, dModel] (linear_pos weight, transposed)
   * @param posU     Learnable bias u [H, d_head]
   * @param posV     Learnable bias v [H, d_head]
   * @returns        Relative position scores [B, H, T, T]
   */
  computeRelativeScores(
    q: TensorHandle,
    posEnc: TensorHandle,
    posProj: TensorHandle,         // [dModel, dModel] — already transposed
    _posU: TensorHandle,            // [H, d_head] — used for AC term in MultiHeadAttention
    posV: TensorHandle,             // [H, d_head]
    numHeads: number,
    headDim: number,
  ): TensorHandle {
    const scope = new ComputeScope();
    const shape = this.backend.getShape(q);
    const B = shape[0] as number;
    const T = shape[2] as number;

    // Project position encoding: [1, 2T-1, dModel] → [1, 2T-1, dModel]
    // then reshape to [1, 2T-1, H, d_head]
    const peFlat = scope.track(this.backend.reshape(posEnc, [2 * T - 1, this.dModel]));
    const peProjFlat = scope.track(this.backend.matmul(peFlat, posProj));  // [2T-1, dModel]
    const peProj = scope.track(this.backend.reshape(peProjFlat, [1, 2 * T - 1, numHeads, headDim]));
    const peProjT = scope.track(this.backend.transpose(peProj, [0, 2, 3, 1]));  // [1, H, d_head, 2T-1]

    // AC term: (Q + u) · K^T  — handled in MultiHeadAttention, not here
    // BD term: (Q + v) · PE^T
    // q: [B, H, T, d_head], posV: [H, d_head] → expand posV to [1, H, 1, d_head]
    const vExp = scope.track(this.backend.reshape(posV, [1, numHeads, 1, headDim]));
    const qv = scope.track(this.backend.add(q, vExp));  // [B, H, T, d_head]

    // qv · peProjT: [B, H, T, d_head] × [1, H, d_head, 2T-1] → [B, H, T, 2T-1]
    const bd = scope.track(this.backend.matmul(qv, peProjT));  // [B, H, T, 2T-1]

    // Shift to get [B, H, T, T] from [B, H, T, 2T-1]
    const shifted = this.relativeShift(bd, B, numHeads, T);
    scope.dispose(this.backend);
    return shifted;
  }

  /**
   * Convert a [B, H, T, 2T-1] relative attention matrix into [B, H, T, T]
   * by selecting the diagonal stripes (the standard "skewing" trick).
   */
  private relativeShift(
    bd: TensorHandle,
    B: number,
    H: number,
    T: number,
  ): TensorHandle {
    const scope = new ComputeScope();

    // Reshape [B, H, T, 2T-1] → [B, H, 2T-1, T] by padding + slicing
    // Standard approach: pad one zero column, reshape, slice
    const padded = scope.track(
      this.backend.pad(bd, [[0, 0], [0, 0], [0, 0], [1, 0]]),
    ); // [B, H, T, 2T]

    const reshaped = scope.track(
      this.backend.reshape(padded, [B, H, 2 * T, T]),
    );

    const sliced = this.backend.slice(reshaped, [0, 0, 1, 0], [B, H, T, T]);
    scope.dispose(this.backend);
    return sliced;
  }
}
