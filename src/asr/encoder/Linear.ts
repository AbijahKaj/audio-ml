import type { ComputeBackend, TensorHandle } from '../compute/index.js';
import type { LinearWeights } from '../model/WeightMapper.js';

/**
 * Linear (fully-connected) layer: y = xW^T + b
 * Weight shape: [out_features, in_features] — PyTorch convention.
 * Input shape:  [B, ..., in_features]
 * Output shape: [B, ..., out_features]
 */
export class Linear {
  constructor(
    private readonly backend: ComputeBackend,
    private readonly weight: TensorHandle,  // [out, in]
    private readonly bias: TensorHandle | null,
  ) {}

  static fromWeights(backend: ComputeBackend, w: LinearWeights): Linear {
    return new Linear(backend, w.weight, w.bias);
  }

  forward(x: TensorHandle): TensorHandle {
    // x: [B, T, in] — matmul expects [..., m, k] × [k, n]
    // weight is [out, in], so we need to transpose it → [in, out]
    const wT = this.backend.transpose(this.weight, [1, 0]);
    let out = this.backend.matmul(x, wT);
    this.backend.dispose(wT);

    if (this.bias) {
      const withBias = this.backend.add(out, this.bias);
      this.backend.dispose(out);
      out = withBias;
    }
    return out;
  }
}
