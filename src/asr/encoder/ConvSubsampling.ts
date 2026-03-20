import type { ComputeBackend } from '../compute/Backend';
import type { TensorHandle } from '../compute/types';
import type { SubsamplingWeights } from '../model/WeightMapper';
import type { FastConformerConfig } from '../model/ModelConfig';

/**
 * Conv2D subsampling for FastConformer.
 * Two stride-2 Conv2D layers reduce time by 4x, plus one more stride-2 to get 8x total.
 * Input: [B, T, mel_bands] -> Output: [B, T/8, d_model]
 *
 * NeMo FastConformer uses "dw_striding" subsampling:
 * - Conv2D (channels, 3x3, stride 2) -> ReLU
 * - Conv2D (channels, 3x3, stride 2) -> ReLU
 * - Linear projection to d_model
 * This gives 4x time downsampling. The 8x comes from an extra stride in subsampling
 * or through pooling at specific layers.
 */
export class ConvSubsampling {
  private backend: ComputeBackend;
  private conv1Weight: TensorHandle;
  private conv1Bias: TensorHandle;
  private conv2Weight: TensorHandle;
  private conv2Bias: TensorHandle;
  private outWeight: TensorHandle;
  private outBias: TensorHandle;
  private dModel: number;

  constructor(backend: ComputeBackend, weights: SubsamplingWeights, config: FastConformerConfig) {
    this.backend = backend;
    this.conv1Weight = weights.conv1Weight;
    this.conv1Bias = weights.conv1Bias;
    this.conv2Weight = weights.conv2Weight;
    this.conv2Bias = weights.conv2Bias;
    this.outWeight = weights.outWeight;
    this.outBias = weights.outBias;
    this.dModel = config.dModel;
  }

  forward(melFeatures: TensorHandle): TensorHandle {
    return this.backend.tidy(() => {
      const shape = this.backend.getShape(melFeatures);
      const B = shape[0] as number;
      const T = shape[1] as number;
      const F = shape[2] as number;

      // [B, T, F] -> [B, T, F, 1] for Conv2D
      let x = this.backend.expandDims(melFeatures, 3);

      // Conv2D 1: stride 2x2 (PyTorch stores as [out, in, H, W], tf needs [H, W, in, out])
      const w1 = this.transposePyTorchConv2d(this.conv1Weight);
      x = this.backend.conv2d(x, w1, [2, 2], 'valid', this.conv1Bias);
      x = this.backend.relu(x);

      // Conv2D 2: stride 2x2
      const w2 = this.transposePyTorchConv2d(this.conv2Weight);
      x = this.backend.conv2d(x, w2, [2, 2], 'valid', this.conv2Bias);
      x = this.backend.relu(x);

      // Flatten spatial dims: [B, T', F', C] -> [B, T', F'*C]
      const newShape = this.backend.getShape(x);
      const T2 = newShape[1] as number;
      const F2 = newShape[2] as number;
      const C = newShape[3] as number;
      x = this.backend.reshape(x, [B, T2, F2 * C]);

      // Linear projection to d_model
      const wOut = this.backend.transpose(this.outWeight, [1, 0]);
      x = this.backend.matmul(x, wOut);
      x = this.backend.add(x, this.outBias);

      return x;
    });
  }

  /**
   * Transpose PyTorch Conv2D weights [out, in, H, W] to TF format [H, W, in, out]
   */
  private transposePyTorchConv2d(weight: TensorHandle): TensorHandle {
    return this.backend.transpose(weight, [2, 3, 1, 0]);
  }
}
