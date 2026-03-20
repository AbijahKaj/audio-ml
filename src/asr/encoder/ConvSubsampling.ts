import * as tf from '@tensorflow/tfjs';
import type { ComputeBackend } from '../compute/Backend';
import type { TensorHandle } from '../compute/types';
import type { SubsamplingWeights } from '../model/WeightMapper';
import type { FastConformerConfig } from '../model/ModelConfig';

/**
 * FastConformer dw_striding subsampling.
 * Conv layers from the actual checkpoint:
 *   conv.0: [256, 1, 3, 3]   — regular Conv2D (1->256, stride=2)
 *   conv.2: [256, 1, 3, 3]   — depthwise Conv2D (groups=256, stride=2)
 *   conv.3: [256, 256, 1, 1] — pointwise Conv2D (256->256, stride=1)
 *   conv.5: [256, 1, 3, 3]   — depthwise Conv2D (groups=256, stride=2)
 *   conv.6: [256, 256, 1, 1] — pointwise Conv2D (256->256, stride=1)
 *   out:    [512, 2560]       — Linear projection
 * Total 8x time downsampling.
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

      // [B, T, F] -> [B, T, F, 1]
      let x = this.backend.expandDims(melFeatures, 3);

      for (let i = 0; i < this.convWeights.length; i++) {
        const w = this.convWeights[i];
        const b = this.convBiases[i];
        const wShape = this.backend.getShape(w);

        // Determine conv type from weight shape
        // [out, in, kH, kW] where in=1 and (kH,kW) != (1,1) → depthwise (or first regular conv)
        // [out, in, 1, 1] → pointwise
        const isPointwise = wShape[2] === 1 && wShape[3] === 1;
        const isDepthwise = wShape[1] === 1 && !isPointwise;

        if (isPointwise) {
          // Pointwise: [C_out, C_in, 1, 1] → TF [1, 1, C_in, C_out], stride 1
          const wTf = this.transposePyTorchConv2d(w);
          x = this.backend.conv2d(x, wTf, [1, 1], 'same', b);
          x = this.backend.relu(x);
        } else if (isDepthwise && i > 0) {
          // Depthwise conv (not first layer): [C, 1, kH, kW]
          // TF depthwiseConv2d expects [kH, kW, C_in, channel_multiplier]
          const wDw = tf.transpose(w as tf.Tensor, [2, 3, 0, 1]); // [kH, kW, C, 1]
          const padH = 1; // for 3x3 kernel
          const padW = 1;
          const padded = tf.pad(x as tf.Tensor, [[0, 0], [padH, padH], [padW, padW], [0, 0]]);
          let result = tf.depthwiseConv2d(
            padded as tf.Tensor4D,
            wDw as tf.Tensor4D,
            [2, 2],
            'valid'
          );
          if (b) {
            result = tf.add(result, b as tf.Tensor) as tf.Tensor4D;
          }
          x = result;
        } else {
          // Regular conv (first layer): [C_out, 1, kH, kW] → [kH, kW, 1, C_out]
          const wTf = this.transposePyTorchConv2d(w);
          x = this.backend.conv2d(x, wTf, [2, 2], 'valid', b);
          x = this.backend.relu(x);
        }
      }

      // Flatten: [B, T', F', C] -> [B, T', F'*C]
      const outShape = this.backend.getShape(x);
      const T2 = outShape[1] as number;
      const F2 = outShape[2] as number;
      const C = outShape[3] as number;
      x = this.backend.reshape(x, [B, T2, F2 * C]);

      // Linear projection to d_model
      const wT = this.backend.transpose(this.outWeight, [1, 0]);
      x = this.backend.matmul(x, wT);
      x = this.backend.add(x, this.outBias);

      return x;
    });
  }

  private transposePyTorchConv2d(weight: TensorHandle): TensorHandle {
    return this.backend.transpose(weight, [2, 3, 1, 0]);
  }
}
