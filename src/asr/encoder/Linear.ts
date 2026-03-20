import type { ComputeBackend } from '../compute/Backend';
import type { TensorHandle } from '../compute/types';
import type { LinearWeightPair } from '../model/WeightMapper';

export class Linear {
  constructor(
    private readonly backend: ComputeBackend,
    private readonly weights: LinearWeightPair,
  ) {}

  forward(input: TensorHandle): TensorHandle {
    if (!this.weights.weight) {
      return input;
    }

    const transposed = this.backend.transpose(this.weights.weight, [1, 0]);
    const projected = this.backend.matmul(input, transposed);
    this.backend.dispose(transposed);

    if (!this.weights.bias) {
      return projected;
    }

    const output = this.backend.add(projected, this.weights.bias);
    this.backend.dispose(projected);
    return output;
  }
}
