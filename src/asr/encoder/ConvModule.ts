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
    this.pw1Weight = weights.pointwise1Weight;
    this.pw1Bias = weights.pointwise1Bias;
    this.depthwiseWeight = weights.depthwiseWeight;
    this.depthwiseBias = weights.depthwiseBias;
    this.bnWeight = weights.batchNorm.weight;
    this.bnBias = weights.batchNorm.bias;
    this.bnMean = weights.batchNorm.runningMean;
    this.bnVar = weights.batchNorm.runningVar;
    this.pw2Weight = weights.pointwise2Weight;
    this.pw2Bias = weights.pointwise2Bias;
    this.kernelSize = kernelSize;
  }

  forward(x: TensorHandle): TensorHandle {
    return this.backend.tidy(() => {
      let h = this.backend.layerNorm(x, this.normWeight, this.normBias, 1e-5);

      // Pointwise conv1 (expand): [B, T, d_model] -> [B, T, 2*d_model]
      h = this.conv1d(h, this.pw1Weight, this.pw1Bias);

      // GLU gating: split into two halves, sigmoid on second
      const parts = this.backend.split(h, 2, -1);
      h = this.backend.mul(parts[0], this.backend.sigmoid(parts[1]));

      // Depthwise conv1d with symmetric padding
      const padding = Math.floor(this.kernelSize / 2);
      h = this.depthwiseConv(h, padding);

      // BatchNorm (inference mode using running stats)
      h = this.backend.batchNorm(h, this.bnMean, this.bnVar, this.bnWeight, this.bnBias, 1e-5);

      // SiLU
      h = this.backend.silu(h);

      // Pointwise conv2 (project back): [B, T, d_model] -> [B, T, d_model]
      h = this.conv1d(h, this.pw2Weight, this.pw2Bias);

      return h;
    });
  }

  forwardStreaming(x: TensorHandle, convState: TensorHandle | null): {
    output: TensorHandle;
    newConvState: TensorHandle;
  } {
    const temps: TensorHandle[] = [];

    let h = this.backend.layerNorm(x, this.normWeight, this.normBias, 1e-5);
    temps.push(h);
    h = this.conv1d(h, this.pw1Weight, this.pw1Bias);
    temps.push(h);

    const parts = this.backend.split(h, 2, -1);
    temps.push(parts[0], parts[1]);
    const sig = this.backend.sigmoid(parts[1]);
    temps.push(sig);
    h = this.backend.mul(parts[0], sig);
    temps.push(h);

    const B = this.backend.getShape(h)[0] as number;
    const chunkLen = this.backend.getShape(x)[1] as number;
    const C = this.backend.getShape(h)[2] as number;
    const stateLen = this.kernelSize - 1;

    const state = convState ?? this.backend.zeros([B, stateLen, C]);
    if (!convState) temps.push(state);
    const concatenated = this.backend.concat([state, h], 1);
    temps.push(concatenated);
    h = concatenated;

    const totalLen = this.backend.getShape(h)[1] as number;

    const newConvState = this.backend.slice(
      h,
      [0, totalLen - stateLen, 0],
      [B, stateLen, C],
    );

    h = this.depthwiseConvNoPad(h);
    temps.push(h);

    const convOutLen = this.backend.getShape(h)[1] as number;
    if (convOutLen > chunkLen) {
      const sliced = this.backend.slice(h, [0, convOutLen - chunkLen, 0], [B, chunkLen, C]);
      temps.push(sliced);
      h = sliced;
    }

    h = this.backend.batchNorm(h, this.bnMean, this.bnVar, this.bnWeight, this.bnBias, 1e-5);
    temps.push(h);
    h = this.backend.silu(h);
    temps.push(h);
    const output = this.conv1d(h, this.pw2Weight, this.pw2Bias);

    for (const t of temps) this.backend.dispose(t);

    return { output, newConvState };
  }

  /**
   * Apply 1D conv using weight stored as [C_out, C_in, kernel_size] or [C_out, C_in].
   * For pointwise (kernel_size=1), this is equivalent to a linear layer.
   */
  private conv1d(x: TensorHandle, weight: TensorHandle, bias: TensorHandle | null): TensorHandle {
    const wShape = this.backend.getShape(weight);

    if (wShape.length === 3 && wShape[2] === 1) {
      // [C_out, C_in, 1] -> treat as linear [C_out, C_in]
      const w2d = this.backend.reshape(weight, [wShape[0], wShape[1]]);
      const wT = this.backend.transpose(w2d, [1, 0]);
      let out = this.backend.matmul(x, wT);
      if (bias) out = this.backend.add(out, bias);
      return out;
    } else if (wShape.length === 2) {
      // [C_out, C_in] -> linear
      const wT = this.backend.transpose(weight, [1, 0]);
      let out = this.backend.matmul(x, wT);
      if (bias) out = this.backend.add(out, bias);
      return out;
    }

    // General conv1d: [C_out, C_in, K] -> need tfjs format [K, C_in, C_out]
    const wTf = this.backend.transpose(weight, [2, 1, 0]);
    const padding = Math.floor((wShape[2] as number) / 2);
    return this.backend.conv1d(x, wTf, 1, padding, bias ?? undefined);
  }

  private depthwiseConv(x: TensorHandle, padding: number): TensorHandle {
    // NeMo: [channels, 1, kernel_size] -> need [kernel_size, channels, 1]
    const wShape = this.backend.getShape(this.depthwiseWeight);
    let w: TensorHandle;

    if (wShape.length === 3) {
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
