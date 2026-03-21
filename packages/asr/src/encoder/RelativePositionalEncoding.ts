import type { ComputeBackend } from '../compute/Backend';
import type { TensorHandle } from '../compute/types';

export class RelativePositionalEncoding {
  private backend: ComputeBackend;
  private dModel: number;

  constructor(backend: ComputeBackend, dModel: number) {
    this.backend = backend;
    this.dModel = dModel;
  }

  /**
   * Generate sinusoidal relative positional encodings.
   * Returns tensor of shape [1, 2*T-1, d_model] for relative positions.
   */
  forward(length: number): TensorHandle {
    if (length <= 0) {
      return this.backend.tensor(new Float32Array(0), [1, 0, this.dModel]);
    }
    return this.backend.tidy(() => {
      const posLen = 2 * length - 1;
      const pe = new Float32Array(posLen * this.dModel);
      const halfDim = this.dModel / 2;

      for (let pos = 0; pos < posLen; pos++) {
        const relPos = pos - (length - 1);
        for (let i = 0; i < halfDim; i++) {
          const angle = relPos / Math.pow(10000, (2 * i) / this.dModel);
          pe[pos * this.dModel + 2 * i] = Math.sin(angle);
          pe[pos * this.dModel + 2 * i + 1] = Math.cos(angle);
        }
      }

      return this.backend.tensor(pe, [1, posLen, this.dModel]);
    });
  }

  /**
   * Compute relative positional attention scores.
   * Q: [B, heads, T, d_head]
   * posEnc: [1, 2T-1, d_model]
   * posProj: projects posEnc to [1, heads, 2T-1, d_head]
   * Returns relative position scores [B, heads, T, T]
   */
  computeRelativeScores(
    query: TensorHandle,
    posEnc: TensorHandle,
    posWeight: TensorHandle,
    numHeads: number,
    headDim: number
  ): TensorHandle {
    return this.backend.tidy(() => {
      const shape = this.backend.getShape(query);
      const T = shape[2] as number;
      const posLen = 2 * T - 1;

      // Project positional encoding: [1, 2T-1, d_model] -> [1, 2T-1, d_model] via weight
      const wT = this.backend.transpose(posWeight, [1, 0]);
      const projectedPos = this.backend.matmul(posEnc, wT); // [1, 2T-1, d_model]

      // Reshape to [1, 2T-1, heads, d_head] then to [1, heads, 2T-1, d_head]
      const reshaped = this.backend.reshape(projectedPos, [1, posLen, numHeads, headDim]);
      const transposed = this.backend.transpose(reshaped, [0, 2, 1, 3]); // [1, heads, 2T-1, d_head]

      // Q * pos^T: [B, heads, T, d_head] x [1, heads, d_head, 2T-1] -> [B, heads, T, 2T-1]
      const posT = this.backend.transpose(transposed, [0, 1, 3, 2]);
      const relScores = this.backend.matmul(query, posT); // [B, heads, T, 2T-1]

      // Skew to convert [B, heads, T, 2T-1] to [B, heads, T, T]
      return this.relativeShift(relScores, T);
    });
  }

  /**
   * Skewing operation: extract the correct relative positions from [B, heads, T, 2T-1]
   * to produce [B, heads, T, T].
   */
  private relativeShift(x: TensorHandle, T: number): TensorHandle {
    return this.backend.tidy(() => {
      const shape = this.backend.getShape(x);
      const B = shape[0] as number;
      const heads = shape[1] as number;

      // Pad left with one column: [B, heads, T, 2T-1] -> [B, heads, T, 2T]
      const padded = this.backend.pad(x, [[0, 0], [0, 0], [0, 0], [1, 0]]);

      // Reshape to [B, heads, 2T, T]
      const reshaped = this.backend.reshape(padded, [B, heads, 2 * T, T]);

      // Slice to get [B, heads, T, T] — take rows 1..T from dim 2
      const sliced = this.backend.slice(reshaped, [0, 0, 1, 0], [B, heads, T, T]);

      return sliced;
    });
  }
}
