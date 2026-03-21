import type { ComputeBackend } from '../compute/Backend';
import type { TensorHandle } from '../compute/types';
import type { SubsamplingWeights } from '../model/WeightMapper';
import type { FastConformerConfig } from '../model/ModelConfig';

/**
 * FastConformer dw_striding subsampling.
 *
 * NeMo processes mel features in [B, 1, F, T] layout (frequency × time) and
 * pads both axes to odd before every stride-2 conv for streaming compatibility.
 * After conv stages: [B, C, F', T'] → permute to [B, T', C, F'] → flatten →
 * [B, T', C*F'] → Linear → [B, T', d_model].
 *
 * Weight format: PyTorch [C_out, C_in, kH, kW] where H=freq, W=time.
 */
export class ConvSubsampling {
  private b: ComputeBackend;
  private convWeights: TensorHandle[];
  private convBiases: TensorHandle[];
  private outWeight: TensorHandle;
  private outBias: TensorHandle;
  private needsOddPad: boolean;

  constructor(backend: ComputeBackend, weights: SubsamplingWeights, config: FastConformerConfig) {
    this.b = backend;
    this.convWeights = weights.allConvWeights;
    this.convBiases = weights.allConvBiases;
    this.outWeight = weights.outWeight;
    this.outBias = weights.outBias;
    // Detect if NeMo's _pad_odd is needed: compare standard conv output dim
    // with the output weight's expected input dim. Models trained with the
    // streaming-compatible ConvSubsampling use _pad_odd (e.g. 128 mel RNNT),
    // while older models don't (e.g. 80 mel TDT).
    const expectedDim = backend.getShape(weights.outWeight)[1] as number;
    const C = backend.getShape(weights.allConvWeights[0])[0] as number;
    let f = config.numMelBands;
    for (let i = 0; i < 3; i++) f = Math.floor((f + 2 - 3) / 2) + 1;
    this.needsOddPad = C * f !== expectedDim;
  }

  forward(melFeatures: TensorHandle): TensorHandle {
    return this.b.tidy(() => {
      const shape = this.b.getShape(melFeatures);
      const B = shape[0] as number;

      let x: TensorHandle;
      if (this.needsOddPad) {
        // Newer NeMo models (e.g. RNNT streaming): [B, 1, F, T] layout
        // with odd-padding before stride-2 convs for streaming compatibility.
        const melFT = this.b.transpose(melFeatures, [0, 2, 1]);
        x = this.b.expandDims(melFT, 1);
      } else {
        // Older NeMo models (e.g. TDT offline): [B, 1, T, F] layout
        x = this.b.expandDims(melFeatures, 1);
      }

      // Layer 0: regular Conv2d(1→C, 3x3, stride=2, pad=1) + ReLU
      x = this.padOddAndConv2d(x, this.convWeights[0], this.convBiases[0], 2, false);
      x = this.b.relu(x);

      // Layer 1: depthwise Conv2d(C, 3x3, stride=2, pad=1)
      x = this.padOddAndConv2d(x, this.convWeights[1], this.convBiases[1], 2, true);

      // Layer 2: pointwise Conv2d(C→C, 1x1) + ReLU
      x = this.conv2dNoPad(x, this.convWeights[2], this.convBiases[2], false);
      x = this.b.relu(x);

      // Layer 3: depthwise Conv2d(C, 3x3, stride=2, pad=1)
      x = this.padOddAndConv2d(x, this.convWeights[3], this.convBiases[3], 2, true);

      // Layer 4: pointwise Conv2d(C→C, 1x1) + ReLU
      x = this.conv2dNoPad(x, this.convWeights[4], this.convBiases[4], false);
      x = this.b.relu(x);

      const xShape = this.b.getShape(x);
      const C = xShape[1] as number;

      let xFlat: TensorHandle;
      if (this.needsOddPad) {
        const Fp = xShape[2] as number;
        const Tp = xShape[3] as number;
        const xPerm = this.b.transpose(x, [0, 3, 1, 2]);
        xFlat = this.b.reshape(xPerm, [B, Tp, C * Fp]);
      } else {
        const Tp = xShape[2] as number;
        const Fp = xShape[3] as number;
        const xPerm = this.b.transpose(x, [0, 2, 1, 3]);
        xFlat = this.b.reshape(xPerm, [B, Tp, C * Fp]);
      }

      // Linear projection → [B, T', d_model]
      const wT = this.b.transpose(this.outWeight, [1, 0]);
      const projected = this.b.matmul(xFlat, wT);
      return this.b.add(projected, this.outBias);
    });
  }

  /**
   * NeMo's _pad_odd: pad frequency and time axes to odd before stride-2 convs,
   * then apply conv2d with padding=1.
   */
  private padOddAndConv2d(
    x: TensorHandle, weight: TensorHandle, bias: TensorHandle,
    stride: number, depthwise: boolean
  ): TensorHandle {
    const xShape = this.b.getShape(x);
    const F = xShape[2] as number;
    const T = xShape[3] as number;
    // NeMo's _pad_odd: pad to odd before stride-2 convs (streaming models only)
    const padF = this.needsOddPad && F % 2 === 0 ? 1 : 0;
    const padT = this.needsOddPad && T % 2 === 0 ? 1 : 0;

    // NCHW → NHWC for TF.js, then add conv padding (1) + odd padding
    let xNhwc = this.b.transpose(x, [0, 2, 3, 1]);
    xNhwc = this.b.pad(xNhwc, [[0, 0], [1, 1 + padF], [1, 1 + padT], [0, 0]]);

    let result;
    if (depthwise) {
      const wDw = this.b.transpose(weight, [2, 3, 0, 1]); // [C,1,kH,kW] → [kH,kW,C,1]
      result = this.b.depthwiseConv2d(xNhwc, wDw, [stride, stride], 'valid', bias);
    } else {
      const wNhwc = this.b.transpose(weight, [2, 3, 1, 0]); // [Cout,Cin,kH,kW] → [kH,kW,Cin,Cout]
      result = this.b.conv2d(xNhwc, wNhwc, [stride, stride], 'valid', bias);
    }

    return this.b.transpose(result, [0, 3, 1, 2]); // NHWC → NCHW
  }

  /** Pointwise (1x1) conv — no padding needed. */
  private conv2dNoPad(
    x: TensorHandle, weight: TensorHandle, bias: TensorHandle, depthwise: boolean
  ): TensorHandle {
    let xNhwc = this.b.transpose(x, [0, 2, 3, 1]);
    let result;
    if (depthwise) {
      const wDw = this.b.transpose(weight, [2, 3, 0, 1]);
      result = this.b.depthwiseConv2d(xNhwc, wDw, [1, 1], 'valid', bias);
    } else {
      const wNhwc = this.b.transpose(weight, [2, 3, 1, 0]);
      result = this.b.conv2d(xNhwc, wNhwc, [1, 1], 'valid', bias);
    }
    return this.b.transpose(result, [0, 3, 1, 2]);
  }
}
