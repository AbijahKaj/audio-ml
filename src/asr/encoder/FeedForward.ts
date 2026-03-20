import type { ComputeBackend, TensorHandle } from '../compute/index.js';
import { ComputeScope } from '../compute/index.js';
import type { FeedForwardWeights } from '../model/WeightMapper.js';
import { Linear } from './Linear.js';

/**
 * Macaron Feed-Forward Module (used twice per ConformerBlock, scaled by 0.5).
 *
 * Forward pass:
 *   x → LayerNorm → Linear(d_model → d_model*expansion) → SiLU → Linear → x
 *
 * Activation: SiLU (a.k.a. Swish), matching NeMo's FastConformer default.
 */
export class FeedForward {
  private readonly norm: { weight: TensorHandle; bias: TensorHandle };
  private readonly fc1: Linear;
  private readonly fc2: Linear;

  constructor(
    private readonly backend: ComputeBackend,
    weights: FeedForwardWeights,
  ) {
    this.norm = weights.norm;
    this.fc1 = Linear.fromWeights(backend, weights.fc1);
    this.fc2 = Linear.fromWeights(backend, weights.fc2);
  }

  forward(x: TensorHandle): TensorHandle {
    const scope = new ComputeScope();

    const normed = scope.track(
      this.backend.layerNorm(x, this.norm.weight, this.norm.bias, 1e-5),
    );
    const hidden = scope.track(this.fc1.forward(normed));
    const activated = scope.track(this.backend.silu(hidden));
    const out = this.fc2.forward(activated);

    scope.dispose(this.backend);
    return out;
  }
}
