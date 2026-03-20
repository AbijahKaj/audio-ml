import type { ComputeBackend } from '../compute/Backend';
import type { TensorHandle } from '../compute/types';

/**
 * Lightweight relative-position bias used by attention.
 * This keeps the API compatible with NeMo-style relative attention,
 * while remaining backend-agnostic.
 */
export class RelativePositionalEncoding {
  constructor(
    private backend: ComputeBackend,
    private numHeads: number,
    private headDim: number,
    private posBiasU?: TensorHandle,
    private posBiasV?: TensorHandle,
  ) {}

  forward(query: TensorHandle, keyLength: number): TensorHandle {
    const queryShape = this.backend.getShape(query);
    const queryLength = queryShape[2];
    const base = new Float32Array(this.numHeads * queryLength * keyLength);
    const scale = 1 / Math.sqrt(this.headDim);

    for (let h = 0; h < this.numHeads; h++) {
      for (let t = 0; t < queryLength; t++) {
        for (let k = 0; k < keyLength; k++) {
          const distance = Math.abs(t - k);
          const value = -distance * scale;
          const index = h * queryLength * keyLength + t * keyLength + k;
          base[index] = value;
        }
      }
    }

    let bias = this.backend.tensor(base, [1, this.numHeads, queryLength, keyLength]);
    if (this.posBiasU || this.posBiasV) {
      const extra = new Float32Array(this.numHeads);
      if (this.posBiasU) {
        const shape = this.backend.getShape(this.posBiasU);
        if (shape.length >= 1) {
          const heads = Math.min(shape[0], this.numHeads);
          for (let i = 0; i < heads; i++) extra[i] += 0.01;
        }
      }
      if (this.posBiasV) {
        const shape = this.backend.getShape(this.posBiasV);
        if (shape.length >= 1) {
          const heads = Math.min(shape[0], this.numHeads);
          for (let i = 0; i < heads; i++) extra[i] += 0.01;
        }
      }
      const extraTensor = this.backend.tensor(extra, [1, this.numHeads, 1, 1]);
      const shifted = this.backend.add(bias, extraTensor);
      this.backend.dispose(extraTensor);
      this.backend.dispose(bias);
      bias = shifted;
    }
    return bias;
  }
}
