import type { ComputeBackend } from '../compute/Backend';
import type { TensorHandle } from '../compute/types';
import type { LayerNormWeights, LinearWeights } from '../model/types';
import { linearForward } from './linear';

export class FeedForward {
  constructor(
    private readonly backend: ComputeBackend,
    private readonly norm: LayerNormWeights,
    private readonly linear1: LinearWeights,
    private readonly linear2: LinearWeights,
    private readonly eps = 1e-5,
  ) {}

  forward(x: TensorHandle): TensorHandle {
    const n = this.backend.layerNorm(x, this.norm.weight, this.norm.bias, this.eps);
    const h = linearForward(this.backend, n, this.linear1);
    this.backend.dispose(n);
    const a = this.backend.silu(h);
    this.backend.dispose(h);
    const out = linearForward(this.backend, a, this.linear2);
    this.backend.dispose(a);
    return out;
  }
}