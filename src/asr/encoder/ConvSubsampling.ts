import type { ComputeBackend, TensorHandle } from '../compute/index.js';
import { ComputeScope } from '../compute/index.js';
import type { SubsamplingWeights } from '../model/WeightMapper.js';
import { Linear } from './Linear.js';

/**
 * Convolutional Subsampling — reduces time dimension by 8x.
 *
 * FastConformer uses two stride-2 Conv2D layers on the mel spectrogram treated
 * as a 2D image [B, T, 80, 1], achieving 4× downsampling on both time and freq.
 * The frequency dimension is then flattened and projected to d_model.
 *
 * Architecture (NeMo ConvSubsampling with subsampling_factor=8):
 *   Input: [B, T, 80]
 *   → reshape [B, T, 80, 1] (H=80, W=T, C=1) — treated as image
 *   → Conv2d(1 → 256, 3×3, stride=2, pad=1) → ReLU → [B, T/2, 40, 256]
 *   → Conv2d(256 → 256, 3×3, stride=2, pad=1) → ReLU → [B, T/4, 20, 256]
 *   → reshape [B, T/4, 20*256]
 *   → Linear(20*256 → d_model)
 *   → [B, T/4, d_model]
 *
 * Note: NeMo's actual 8× factor for FastConformer uses a slightly different
 * conv setup. The exact shape depends on the checkpoint. The subsampling weight
 * shape from the checkpoint tells us the actual channel/stride config.
 */
export class ConvSubsampling {
  private readonly conv1Weight: TensorHandle;  // [C_out, C_in, kH, kW] PyTorch → need to adapt
  private readonly conv1Bias: TensorHandle;
  private readonly conv2Weight: TensorHandle;
  private readonly conv2Bias: TensorHandle;
  private readonly outLinear: Linear;
  private readonly convChannels: number;

  constructor(
    private readonly backend: ComputeBackend,
    weights: SubsamplingWeights,
    _dModel: number,
    _numMelBands: number,
    convChannels: number,
  ) {
    this.convChannels = convChannels;
    void _dModel;

    // PyTorch Conv2d weight: [C_out, C_in, kH, kW]
    // TF.js conv2d kernel: [kH, kW, C_in, C_out]
    this.conv1Weight = this.adaptConvWeight(backend, weights.conv1.weight);
    this.conv1Bias = weights.conv1.bias;
    this.conv2Weight = this.adaptConvWeight(backend, weights.conv2.weight);
    this.conv2Bias = weights.conv2.bias;
    this.outLinear = Linear.fromWeights(backend, weights.out);
  }

  /**
   * Convert PyTorch Conv2d weight [C_out, C_in, kH, kW] → TF.js [kH, kW, C_in, C_out]
   */
  private adaptConvWeight(
    backend: ComputeBackend,
    w: TensorHandle,
  ): TensorHandle {
    // w shape: [C_out, C_in, kH, kW] → transpose to [kH, kW, C_in, C_out]
    return backend.transpose(w, [2, 3, 1, 0]);
  }

  forward(melFeatures: TensorHandle): TensorHandle {
    const scope = new ComputeScope();
    const shape = this.backend.getShape(melFeatures);
    const B = shape[0] as number;

    // NeMo's ConvSubsampling processes mel features as [B, 1, T, 80] (channels first)
    // then applies Conv2d. We work in channels-last [B, T, 80, 1] (TF.js convention).

    // [B, T, 80] → [B, T, 80, 1]
    const x4d = scope.track(this.backend.expandDims(melFeatures, 3));

    // Conv2d 1: 1 → convChannels, stride=2×2, pad=same
    const c1Raw = scope.track(
      this.backend.conv2d(x4d, this.conv1Weight, null, [2, 2], 'same'),
    );
    const c1Bias = scope.track(
      this.backend.add(c1Raw, this.backend.reshape(this.conv1Bias, [1, 1, 1, this.convChannels])),
    );
    const c1 = scope.track(this.backend.relu(c1Bias));
    // Shape: [B, T/2, 40, convChannels]

    // Conv2d 2: convChannels → convChannels, stride=2×2, pad=same
    const c2Raw = scope.track(
      this.backend.conv2d(c1, this.conv2Weight, null, [2, 2], 'same'),
    );
    const c2Bias = scope.track(
      this.backend.add(c2Raw, this.backend.reshape(this.conv2Bias, [1, 1, 1, this.convChannels])),
    );
    const c2 = scope.track(this.backend.relu(c2Bias));
    // Shape: [B, T/4, 20, convChannels]

    const c2Shape = this.backend.getShape(c2);
    const T4 = c2Shape[1] as number;
    const F4 = c2Shape[2] as number;

    // Flatten freq × channels: [B, T/4, 20, convChannels] → [B, T/4, 20*convChannels]
    const flat = scope.track(
      this.backend.reshape(c2, [B, T4, F4 * this.convChannels]),
    );

    // Linear projection to d_model
    const out = this.outLinear.forward(flat);

    scope.dispose(this.backend);
    return out;
  }
}
