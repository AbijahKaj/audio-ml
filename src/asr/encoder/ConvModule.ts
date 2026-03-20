import type { ComputeBackend, TensorHandle } from '../compute/index.js';
import { ComputeScope } from '../compute/index.js';
import type { ConvModuleWeights } from '../model/WeightMapper.js';

/**
 * Convolution Module in each ConformerBlock.
 *
 * Forward pass (NeMo ConvolutionModule):
 *   x → LayerNorm
 *     → Pointwise Conv (d_model → 2*d_model) + GLU gate (sigmoid of second half)
 *     → Depthwise Conv1D (kernel_size=9, groups=d_model)
 *     → BatchNorm
 *     → SiLU
 *     → Pointwise Conv (d_model → d_model)
 *
 * Input/output: [B, T, d_model]
 */
export class ConvModule {
  private readonly norm: { weight: TensorHandle; bias: TensorHandle };
  private readonly pw1Weight: TensorHandle;
  private readonly pw1Bias: TensorHandle;
  private readonly dwWeight: TensorHandle;
  private readonly bnWeight: TensorHandle;
  private readonly bnBias: TensorHandle;
  private readonly bnMean: TensorHandle;
  private readonly bnVar: TensorHandle;
  private readonly pw2Weight: TensorHandle;
  private readonly pw2Bias: TensorHandle;
  private readonly dModel: number;
  private readonly kernelSize: number;

  constructor(
    private readonly backend: ComputeBackend,
    weights: ConvModuleWeights,
    dModel: number,
    kernelSize: number,
  ) {
    this.norm = weights.norm;
    this.pw1Weight = weights.pointwiseConv1.weight;
    this.pw1Bias = weights.pointwiseConv1.bias;
    this.dwWeight = weights.depthwiseConv.weight;
    this.bnWeight = weights.batchNorm.weight;
    this.bnBias = weights.batchNorm.bias;
    this.bnMean = weights.batchNorm.mean;
    this.bnVar = weights.batchNorm.var;
    this.pw2Weight = weights.pointwiseConv2.weight;
    this.pw2Bias = weights.pointwiseConv2.bias;
    this.dModel = dModel;
    this.kernelSize = kernelSize;
  }

  forward(x: TensorHandle): TensorHandle {
    const scope = new ComputeScope();
    // 1. Layer norm
    const normed = scope.track(
      this.backend.layerNorm(x, this.norm.weight, this.norm.bias, 1e-5),
    );

    // 2. Pointwise conv 1: d_model → 2*d_model
    // Weight in NeMo: [2*d_model, d_model, 1] (Conv1d weight) → we treat as Linear
    // NeMo stores pointwise_conv as Conv1d with kernel=1, but effectively it's Linear.
    // We reshape weight from [2*dModel, dModel, 1] → [2*dModel, dModel] to do matmul.
    const pw1W = scope.track(this.backend.reshape(
      this.pw1Weight, [2 * this.dModel, this.dModel],
    ));
    const pw1WT = scope.track(this.backend.transpose(pw1W, [1, 0]));
    const pw1Out = scope.track(this.backend.matmul(normed, pw1WT));
    const pw1WithBias = scope.track(this.backend.add(pw1Out, this.pw1Bias));

    // 3. GLU gate: split in half, apply sigmoid to second half
    const [a, b] = this.backend.split(pw1WithBias, 2, -1);
    const aT = scope.track(a);
    const bT = scope.track(b);
    const gate = scope.track(this.backend.sigmoid(bT));
    let h = scope.track(this.backend.mul(aT, gate));  // [B, T, d_model]

    // 4. Depthwise conv1d (kernel_size, d_model)
    // NeMo weight shape: [d_model, 1, kernel_size] (groups=d_model Conv1d)
    // Reshape to [kernel_size, d_model] for our depthwiseConv1d which expects [K, C]
    const dwW = scope.track(this.backend.reshape(
      this.dwWeight, [this.kernelSize, this.dModel],
    ));
    h = scope.track(this.backend.depthwiseConv1d(h, dwW, 1, 'same'));

    // 5. Batch norm (applied along the channel dimension)
    // BN stats are [d_model], need to broadcast over [B, T, d_model]
    h = scope.track(this.backend.batchNorm(
      h, this.bnMean, this.bnVar, this.bnWeight, this.bnBias, 1e-5,
    ));

    // 6. SiLU
    h = scope.track(this.backend.silu(h));

    // 7. Pointwise conv 2: d_model → d_model
    const pw2W = scope.track(this.backend.reshape(
      this.pw2Weight, [this.dModel, this.dModel],
    ));
    const pw2WT = scope.track(this.backend.transpose(pw2W, [1, 0]));
    const pw2Out = scope.track(this.backend.matmul(h, pw2WT));
    const out = this.backend.add(pw2Out, this.pw2Bias);

    scope.dispose(this.backend);
    return out;
  }
}
