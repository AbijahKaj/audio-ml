import * as tf from '@tensorflow/tfjs';
import type { ComputeBackend } from '../compute/Backend';
import type { TensorHandle } from '../compute/types';
import type { SubsamplingWeights } from '../model/WeightMapper';
import type { FastConformerConfig } from '../model/ModelConfig';

/**
 * FastConformer dw_striding subsampling.
 * NeMo architecture:
 *   [0] Conv2d(1, 256, 3x3, stride=2, padding=1) → ReLU
 *   [1] Conv2d(256, 256, 3x3, stride=2, padding=1, groups=256) depthwise
 *   [2] Conv2d(256, 256, 1x1) pointwise → ReLU
 *   [3] Conv2d(256, 256, 3x3, stride=2, padding=1, groups=256) depthwise
 *   [4] Conv2d(256, 256, 1x1) pointwise → ReLU
 *   Linear(C*F', d_model)
 *
 * Weight indices from checkpoint:
 *   0: [256, 1, 3, 3]  — regular conv (1→256)
 *   1: [256, 1, 3, 3]  — depthwise (groups=256)
 *   2: [256, 256, 1, 1] — pointwise
 *   3: [256, 1, 3, 3]  — depthwise
 *   4: [256, 256, 1, 1] — pointwise
 */
export class ConvSubsampling {
  private backend: ComputeBackend;
  private convWeights: TensorHandle[];
  private convBiases: TensorHandle[];
  private outWeight: TensorHandle;
  private outBias: TensorHandle;

  constructor(backend: ComputeBackend, weights: SubsamplingWeights, _config: FastConformerConfig) {
    this.backend = backend;
    this.convWeights = weights.allConvWeights;
    this.convBiases = weights.allConvBiases;
    this.outWeight = weights.outWeight;
    this.outBias = weights.outBias;
  }

  forward(melFeatures: TensorHandle): TensorHandle {
    return (tf.tidy as Function)(() => {
      const shape = this.backend.getShape(melFeatures);
      const B = shape[0] as number;

      // [B, T, F] → [B, 1, T, F] (NeMo puts time on H, freq on W)
      let xNchw: tf.Tensor = (melFeatures as tf.Tensor).expandDims(1); // [B, 1, T, F]

      // Layer 0: regular Conv2d(1→256, 3x3, stride=2, pad=1) + ReLU
      xNchw = this.conv2dWithPad(xNchw, this.convWeights[0], this.convBiases[0], 2, 1, false);
      xNchw = tf.relu(xNchw);

      // Layer 1: depthwise Conv2d(256, 3x3, stride=2, pad=1, groups=256)
      xNchw = this.depthwiseConv2dWithPad(xNchw, this.convWeights[1], this.convBiases[1], 2, 1);

      // Layer 2: pointwise Conv2d(256→256, 1x1) + ReLU
      xNchw = this.conv2dWithPad(xNchw, this.convWeights[2], this.convBiases[2], 1, 0, false);
      xNchw = tf.relu(xNchw);

      // Layer 3: depthwise Conv2d(256, 3x3, stride=2, pad=1, groups=256)
      xNchw = this.depthwiseConv2dWithPad(xNchw, this.convWeights[3], this.convBiases[3], 2, 1);

      // Layer 4: pointwise Conv2d(256→256, 1x1) + ReLU
      xNchw = this.conv2dWithPad(xNchw, this.convWeights[4], this.convBiases[4], 1, 0, false);
      xNchw = tf.relu(xNchw);

      // xNchw: [B, C, T', F'] where T' = downsampled time, F' = downsampled freq
      const xShape = xNchw.shape;
      const C = xShape[1] as number;
      const Tp = xShape[2] as number;
      const Fp = xShape[3] as number;

      // NeMo: x.transpose(1,2).reshape(b,t,-1) → [B, T', C*F']
      const xPerm = tf.transpose(xNchw, [0, 2, 1, 3]); // [B, T', C, F']
      const xFlat = tf.reshape(xPerm, [B, Tp, C * Fp]);

      // Linear projection to d_model
      const wT = tf.transpose(this.outWeight as tf.Tensor, [1, 0]);
      let out = tf.matMul(xFlat, wT);
      out = tf.add(out, this.outBias as tf.Tensor) as tf.Tensor;

      return out; // [B, T', d_model]
    });
  }

  private conv2dWithPad(
    xNchw: tf.Tensor, weight: TensorHandle, bias: TensorHandle,
    stride: number, pad: number, _depthwise: boolean
  ): tf.Tensor {
    const w = weight as tf.Tensor;
    // PyTorch [C_out, C_in, kH, kW] → TF NHWC [kH, kW, C_in, C_out]
    const wNhwc = tf.transpose(w, [2, 3, 1, 0]);
    let xNhwc = tf.transpose(xNchw, [0, 2, 3, 1]); // [B, H, W, C]
    if (pad > 0) {
      xNhwc = tf.pad(xNhwc, [[0, 0], [pad, pad], [pad, pad], [0, 0]]);
    }
    let result = tf.conv2d(xNhwc as tf.Tensor4D, wNhwc as tf.Tensor4D, [stride, stride], 'valid');
    if (bias) result = tf.add(result, bias as tf.Tensor) as tf.Tensor4D;
    return tf.transpose(result, [0, 3, 1, 2]); // back to NCHW
  }

  private depthwiseConv2dWithPad(
    xNchw: tf.Tensor, weight: TensorHandle, bias: TensorHandle,
    stride: number, pad: number
  ): tf.Tensor {
    const w = weight as tf.Tensor;
    // PyTorch depthwise [C, 1, kH, kW] → TF [kH, kW, C, 1]
    const wDw = tf.transpose(w, [2, 3, 0, 1]);
    let xNhwc = tf.transpose(xNchw, [0, 2, 3, 1]); // [B, H, W, C]
    if (pad > 0) {
      xNhwc = tf.pad(xNhwc, [[0, 0], [pad, pad], [pad, pad], [0, 0]]);
    }
    let result = tf.depthwiseConv2d(xNhwc as tf.Tensor4D, wDw as tf.Tensor4D, [stride, stride], 'valid');
    if (bias) result = tf.add(result, bias as tf.Tensor) as tf.Tensor4D;
    return tf.transpose(result, [0, 3, 1, 2]); // back to NCHW
  }
}
