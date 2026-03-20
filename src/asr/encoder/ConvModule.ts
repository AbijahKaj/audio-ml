import type { ComputeBackend } from '../compute/Backend';
import type { TensorHandle } from '../compute/types';
import type { ConvModuleWeights } from '../model/WeightMapper';

export class ConvModule {
  private backend: ComputeBackend;
  private normWeight: TensorHandle;
  private normBias: TensorHandle;
  private pw1Weight: TensorHandle;
  private pw1Bias: TensorHandle | null;
  private depthwiseWeight: TensorHandle;
  private depthwiseBias: TensorHandle | null;
  private bnWeight: TensorHandle;
  private bnBias: TensorHandle;
  private bnMean: TensorHandle;
  private bnVar: TensorHandle;
  private pw2Weight: TensorHandle;
  private pw2Bias: TensorHandle | null;
  private kernelSize: number;

  constructor(backend: ComputeBackend, weights: ConvModuleWeights, kernelSize: number) {
    this.backend = backend;
    this.normWeight = weights.norm.weight;
    this.normBias = weights.norm.bias;
    this.pw1Weight = weights.pointwise1.weight;
    this.pw1Bias = weights.pointwise1.bias;
    this.depthwiseWeight = weights.depthwiseWeight;
    this.depthwiseBias = weights.depthwiseBias;
    this.bnWeight = weights.batchNorm.weight;
    this.bnBias = weights.batchNorm.bias;
    this.bnMean = weights.batchNorm.runningMean;
    this.bnVar = weights.batchNorm.runningVar;
    this.pw2Weight = weights.pointwise2.weight;
    this.pw2Bias = weights.pointwise2.bias;
    this.kernelSize = kernelSize;
  }

  forward(x: TensorHandle): TensorHandle {
    return this.backend.tidy(() => {
      let h = this.backend.layerNorm(x, this.normWeight, this.normBias, 1e-5);

      // Pointwise conv1 (expand): [B, T, d_model] -> [B, T, 2*d_model]
      h = this.pointwiseConv(h, this.pw1Weight, this.pw1Bias);

      // GLU gating
      const parts = this.backend.split(h, 2, -1);
      h = this.backend.mul(parts[0], this.backend.sigmoid(parts[1]));

      // Depthwise conv1d with causal-like padding
      const padding = Math.floor(this.kernelSize / 2);
      h = this.depthwiseConv(h, padding);

      // BatchNorm
      h = this.backend.batchNorm(h, this.bnMean, this.bnVar, this.bnWeight, this.bnBias, 1e-5);

      // SiLU
      h = this.backend.silu(h);

      // Pointwise conv2 (project back): [B, T, d_model] -> [B, T, d_model]
      h = this.pointwiseConv(h, this.pw2Weight, this.pw2Bias);

      return h;
    });
  }

  forwardStreaming(x: TensorHandle, convState: TensorHandle | null): {
    output: TensorHandle;
    newConvState: TensorHandle;
  } {
    let h = this.backend.layerNorm(x, this.normWeight, this.normBias, 1e-5);

    h = this.pointwiseConv(h, this.pw1Weight, this.pw1Bias);

    const parts = this.backend.split(h, 2, -1);
    h = this.backend.mul(parts[0], this.backend.sigmoid(parts[1]));

    // For streaming, prepend cached conv state
    if (convState) {
      h = this.backend.concat([convState, h], 1);
    }

    const totalLen = this.backend.getShape(h)[1] as number;
    const chunkLen = this.backend.getShape(x)[1] as number;
    const stateLen = this.kernelSize - 1;

    // Save state for next chunk
    const newStateStart = Math.max(0, totalLen - stateLen);
    const newConvState = this.backend.slice(
      h,
      [0, newStateStart, 0],
      [this.backend.getShape(h)[0] as number, Math.min(stateLen, totalLen), this.backend.getShape(h)[2] as number]
    );

    // Apply depthwise conv without extra padding (state provides left context)
    h = this.depthwiseConvNoPad(h);

    // Trim to chunk length
    const convOutLen = this.backend.getShape(h)[1] as number;
    if (convOutLen > chunkLen) {
      h = this.backend.slice(h, [0, convOutLen - chunkLen, 0], [
        this.backend.getShape(h)[0] as number,
        chunkLen,
        this.backend.getShape(h)[2] as number
      ]);
    }

    h = this.backend.batchNorm(h, this.bnMean, this.bnVar, this.bnWeight, this.bnBias, 1e-5);
    h = this.backend.silu(h);
    h = this.pointwiseConv(h, this.pw2Weight, this.pw2Bias);

    return { output: h, newConvState };
  }

  private pointwiseConv(x: TensorHandle, weight: TensorHandle, bias: TensorHandle | null): TensorHandle {
    // Pointwise is just a linear: [B, T, C_in] x [C_out, C_in]^T -> [B, T, C_out]
    const wT = this.backend.transpose(weight, [1, 0]);
    let out = this.backend.matmul(x, wT);
    if (bias) out = this.backend.add(out, bias);
    return out;
  }

  private depthwiseConv(x: TensorHandle, padding: number): TensorHandle {
    // NeMo stores depthwise conv weight as [channels, 1, kernel_size]
    // We need [kernel_size, channels, 1] for depthwiseConv1d
    const wShape = this.backend.getShape(this.depthwiseWeight);
    let w: TensorHandle;

    if (wShape.length === 3) {
      // [channels, 1, kernel_size] -> [kernel_size, channels, 1]
      w = this.backend.transpose(this.depthwiseWeight, [2, 0, 1]);
    } else {
      w = this.depthwiseWeight;
    }

    let h = this.backend.depthwiseConv1d(x, w, 1, padding);

    if (this.depthwiseBias) {
      h = this.backend.add(h, this.depthwiseBias);
    }

    return h;
  }

  private depthwiseConvNoPad(x: TensorHandle): TensorHandle {
    const wShape = this.backend.getShape(this.depthwiseWeight);
    let w: TensorHandle;

    if (wShape.length === 3) {
      w = this.backend.transpose(this.depthwiseWeight, [2, 0, 1]);
    } else {
      w = this.depthwiseWeight;
    }

    let h = this.backend.depthwiseConv1d(x, w, 1, 0);

    if (this.depthwiseBias) {
      h = this.backend.add(h, this.depthwiseBias);
    }

    return h;
  }
}
