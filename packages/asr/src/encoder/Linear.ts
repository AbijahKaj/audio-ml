import type { ComputeBackend } from '../compute/Backend';
import type { TensorHandle } from '../compute/types';
import type { LinearWeights } from '../model/WeightMapper';

export class Linear {
  private backend: ComputeBackend;
  private weight: TensorHandle;
  private bias: TensorHandle | null;

  constructor(backend: ComputeBackend, weights: LinearWeights) {
    this.backend = backend;
    this.weight = weights.weight;
    this.bias = weights.bias;
  }

  forward(x: TensorHandle): TensorHandle {
    return this.backend.tidy(() => {
      const wT = this.backend.transpose(this.weight, [1, 0]);
      const out = this.backend.matmul(x, wT);
      if (this.bias) {
        return this.backend.add(out, this.bias);
      }
      return out;
    });
  }
}
