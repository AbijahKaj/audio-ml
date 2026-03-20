import type { ComputeBackend } from '../compute/Backend';
import type { TensorHandle } from '../compute/types';
import type { SubsamplingWeights } from '../model/WeightMapper';
import type { FastConformerConfig } from '../model/ModelConfig';

/**
 * FastConformer dw_striding subsampling.
 * NeMo architecture:
 *   [0] Conv2d(1, 256, 3x3, stride=2, padding=1) + ReLU
 *   [1] Conv2d(256, 256, 3x3, stride=2, padding=1, groups=256) depthwise
 *   [2] Conv2d(256, 256, 1x1) pointwise + ReLU
 *   [3] Conv2d(256, 256, 3x3, stride=2, padding=1, groups=256) depthwise
 *   [4] Conv2d(256, 256, 1x1) pointwise + ReLU
 *   Linear(C*F', d_model)
 *
 * All ops go through ComputeBackend — no direct tf.* imports.
 * Weights arrive as PyTorch layout and are transposed to NHWC here.
 */
export class ConvSubsampling {
  private b: ComputeBackend;
  private convWeights: TensorHandle[];
  private convBiases: TensorHandle[];
  private outWeight: TensorHandle;
  private outBias: TensorHandle;

  constructor(backend: ComputeBackend, weights: SubsamplingWeights, _config: FastConformerConfig) {
    this.b = backend;
    this.convWeights = weights.allConvWeights;
    this.convBiases = weights.allConvBiases;
    this.outWeight = weights.outWeight;
    this.outBias = weights.outBias;
  }

  forward(melFeatures: TensorHandle): TensorHandle {
    return this.b.tidy(() => {
      const shape = this.b.getShape(melFeatures);
      const B = shape[0] as number;

      // [B, T, F] → [B, 1, T, F]  (channel dim for conv2d)
      let xNchw = this.b.expandDims(melFeatures, 1);

      // Layer 0: regular Conv2d(1→256, 3x3, stride=2, pad=1) + ReLU
      xNchw = this.conv2dWithPad(xNchw, this.convWeights[0], this.convBiases[0], 2, 1, false);
      xNchw = this.b.relu(xNchw);

      // Layer 1: depthwise Conv2d(256, 3x3, stride=2, pad=1, groups=256)
      xNchw = this.depthwiseConv2dWithPad(xNchw, this.convWeights[1], this.convBiases[1], 2, 1);

      // Layer 2: pointwise Conv2d(256→256, 1x1) + ReLU
      xNchw = this.conv2dWithPad(xNchw, this.convWeights[2], this.convBiases[2], 1, 0, false);
      xNchw = this.b.relu(xNchw);

      // Layer 3: depthwise Conv2d(256, 3x3, stride=2, pad=1, groups=256)
      xNchw = this.depthwiseConv2dWithPad(xNchw, this.convWeights[3], this.convBiases[3], 2, 1);

      // Layer 4: pointwise Conv2d(256→256, 1x1) + ReLU
      xNchw = this.conv2dWithPad(xNchw, this.convWeights[4], this.convBiases[4], 1, 0, false);
      xNchw = this.b.relu(xNchw);

      // xNchw: [B, C, T', F']
      const xShape = this.b.getShape(xNchw);
      const C = xShape[1] as number;
      const Tp = xShape[2] as number;
      const Fp = xShape[3] as number;

      // NeMo: x.transpose(1,2).reshape(b,t,-1) → [B, T', C*F']
      const xPerm = this.b.transpose(xNchw, [0, 2, 1, 3]);
      const xFlat = this.b.reshape(xPerm, [B, Tp, C * Fp]);

      // Linear projection to d_model
      const wT = this.b.transpose(this.outWeight, [1, 0]);
      const projected = this.b.matmul(xFlat, wT);
      return this.b.add(projected, this.outBias);
    });
  }

  /** Regular conv2d: PyTorch [C_out, C_in, kH, kW] → NHWC [kH, kW, C_in, C_out] */
  private conv2dWithPad(
    xNchw: TensorHandle, weight: TensorHandle, bias: TensorHandle,
    stride: number, pad: number, _depthwise: boolean
  ): TensorHandle {
    const wNhwc = this.b.transpose(weight, [2, 3, 1, 0]);
    let xNhwc = this.b.transpose(xNchw, [0, 2, 3, 1]);
    if (pad > 0) {
      xNhwc = this.b.pad(xNhwc, [[0, 0], [pad, pad], [pad, pad], [0, 0]]);
    }
    const result = this.b.conv2d(xNhwc, wNhwc, [stride, stride], 'valid', bias);
    return this.b.transpose(result, [0, 3, 1, 2]);
  }

  /** Depthwise conv2d: PyTorch [C, 1, kH, kW] → NHWC [kH, kW, C, 1] */
  private depthwiseConv2dWithPad(
    xNchw: TensorHandle, weight: TensorHandle, bias: TensorHandle,
    stride: number, pad: number
  ): TensorHandle {
    const wDw = this.b.transpose(weight, [2, 3, 0, 1]);
    let xNhwc = this.b.transpose(xNchw, [0, 2, 3, 1]);
    if (pad > 0) {
      xNhwc = this.b.pad(xNhwc, [[0, 0], [pad, pad], [pad, pad], [0, 0]]);
    }
    const result = this.b.depthwiseConv2d(xNhwc, wDw, [stride, stride], 'valid', bias);
    return this.b.transpose(result, [0, 3, 1, 2]);
  }
}
