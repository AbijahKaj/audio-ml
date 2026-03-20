import type { ComputeBackend } from '../compute/Backend';
import type { ConformerLayerWeights } from '../model/weights';
import type { TensorHandle } from '../compute/types';
import { linearForward } from './ops';

export class FeedForward {
  constructor(
    private readonly backend: ComputeBackend,
    private readonly linear1: ConformerLayerWeights['feedForward1']['linear1'],
    private readonly linear2: ConformerLayerWeights['feedForward1']['linear2'],
  ) {}

  forward(x: TensorHandle): TensorHandle {
    const b = this.backend;
    let h = linearForward(b, x, this.linear1.weight, this.linear1.bias);
    h = b.silu(h);
    const out = linearForward(b, h, this.linear2.weight, this.linear2.bias);
    b.dispose(h);
    return out;
  }
}
