import { ComputeScope } from '../compute/ComputeScope';
import type { ComputeBackend } from '../compute/Backend';
import type { TensorHandle } from '../compute/types';
import type { FeedForwardWeights } from '../model/ModelWeights';
import { Linear } from './Linear';

export class FeedForward {
  private linear1: Linear;
  private linear2: Linear;

  constructor(
    private backend: ComputeBackend,
    private weights: FeedForwardWeights,
  ) {
    this.linear1 = new Linear(backend, weights.linear1);
    this.linear2 = new Linear(backend, weights.linear2);
  }

  forward(x: TensorHandle): TensorHandle {
    const scope = new ComputeScope();
    const normed = scope.track(
      this.backend.layerNorm(
        x,
        this.weights.norm.weight,
        this.weights.norm.bias,
        1e-5,
      ),
    );
    const hidden = scope.track(this.linear1.forward(normed));
    const activated = scope.track(this.backend.silu(hidden));
    const output = this.linear2.forward(activated);
    scope.keep(output);
    scope.dispose(this.backend);
    return output;
  }
}
