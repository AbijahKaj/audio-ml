import { ComputeScope } from '../compute/Scope.js';
import type { ComputeBackend } from '../compute/Backend.js';
import type { TensorHandle } from '../compute/types.js';
import type { FeedForwardWeights } from '../model/WeightMapper.js';
import { Linear } from './Linear.js';

export class FeedForward {
  private readonly linear1: Linear;
  private readonly linear2: Linear;

  constructor(
    private readonly backend: ComputeBackend,
    private readonly weights: FeedForwardWeights,
  ) {
    this.linear1 = new Linear(backend, weights.linear1);
    this.linear2 = new Linear(backend, weights.linear2);
  }

  forward(input: TensorHandle): TensorHandle {
    if (!this.weights.linear1.weight || !this.weights.linear2.weight) {
      return input;
    }

    const scope = new ComputeScope();
    const normalized = this.weights.norm.weight && this.weights.norm.bias
      ? scope.track(this.backend.layerNorm(input, this.weights.norm.weight, this.weights.norm.bias, 1e-5))
      : input;
    const hidden = scope.track(this.linear1.forward(normalized));
    const activated = scope.track(this.backend.silu(hidden));
    const output = this.linear2.forward(activated);
    scope.dispose(this.backend);
    return output;
  }
}
