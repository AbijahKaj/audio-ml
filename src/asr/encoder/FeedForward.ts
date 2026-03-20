import type { ComputeBackend } from '../compute/Backend';
import type { TensorHandle } from '../compute/types';
import type { FeedForwardWeights } from '../model/WeightMapper';
import { Linear } from './Linear';

export class FeedForward {
  private backend: ComputeBackend;
  private normWeight: TensorHandle;
  private normBias: TensorHandle;
  private linear1: Linear;
  private linear2: Linear;

  constructor(backend: ComputeBackend, weights: FeedForwardWeights) {
    this.backend = backend;
    this.normWeight = weights.norm.weight;
    this.normBias = weights.norm.bias;
    this.linear1 = new Linear(backend, weights.linear1);
    this.linear2 = new Linear(backend, weights.linear2);
  }

  forward(x: TensorHandle): TensorHandle {
    return this.backend.tidy(() => {
      let h = this.backend.layerNorm(x, this.normWeight, this.normBias, 1e-5);
      h = this.linear1.forward(h);
      h = this.backend.silu(h);
      h = this.linear2.forward(h);
      return h;
    });
  }
}
