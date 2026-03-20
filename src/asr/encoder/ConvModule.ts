import { ComputeScope } from '../compute/ComputeScope';
import type { ComputeBackend } from '../compute/Backend';
import type { TensorHandle } from '../compute/types';
import type { ConvModuleWeights } from '../model/ModelWeights';
import { Linear } from './Linear';

export interface StreamingConvOutput {
  output: TensorHandle;
  convState: TensorHandle;
}

export class ConvModule {
  private readonly pointwise1: Linear;
  private readonly pointwise2: Linear;

  constructor(
    private backend: ComputeBackend,
    private weights: ConvModuleWeights,
  ) {
    this.pointwise1 = new Linear(backend, weights.pointwise1);
    this.pointwise2 = new Linear(backend, weights.pointwise2);
  }

  forward(x: TensorHandle): TensorHandle {
    const scope = new ComputeScope();
    const normed = scope.track(this.backend.layerNorm(x, this.weights.norm.weight, this.weights.norm.bias, 1e-5));
    const expanded = scope.track(this.pointwise1.forward(normed));
    const gated = scope.track(this.glu(expanded));
    const depthwise = scope.track(this.depthwise(gated));
    const bn = scope.track(this.applyBatchNormOrAffine(depthwise));
    const activated = scope.track(this.backend.silu(bn));
    const output = this.pointwise2.forward(activated);
    scope.keep(output);
    scope.dispose(this.backend);
    return output;
  }

  forwardStreaming(chunk: TensorHandle, previousConvState?: TensorHandle): StreamingConvOutput {
    const scope = new ComputeScope();
    const chunkShape = this.backend.getShape(chunk);
    const chunkLength = chunkShape[1];
    const normed = scope.track(
      this.backend.layerNorm(chunk, this.weights.norm.weight, this.weights.norm.bias, 1e-5),
    );
    const expanded = scope.track(this.pointwise1.forward(normed));
    const gatedChunk = scope.track(this.glu(expanded));
    const fullInput = previousConvState
      ? scope.track(this.backend.concat([previousConvState, gatedChunk], 1))
      : gatedChunk;

    const depthwise = scope.track(this.depthwise(fullInput));
    const depthShape = this.backend.getShape(depthwise);
    const offset = Math.max(0, depthShape[1] - chunkLength);
    const chunkOnly = scope.track(
      this.backend.slice(depthwise, [0, offset, 0], [depthShape[0], chunkLength, depthShape[2]]),
    );
    const bn = scope.track(this.applyBatchNormOrAffine(chunkOnly));
    const activated = scope.track(this.backend.silu(bn));
    const output = this.pointwise2.forward(activated);
    const newState = this.extractConvState(fullInput);

    scope.keep(output);
    scope.keep(newState);
    scope.dispose(this.backend);

    return { output, convState: newState };
  }

  private glu(x: TensorHandle): TensorHandle {
    const [a, b] = this.backend.split(x, 2, -1);
    const gate = this.backend.sigmoid(b);
    const out = this.backend.mul(a, gate);
    this.backend.dispose(a);
    this.backend.dispose(b);
    this.backend.dispose(gate);
    return out;
  }

  private depthwise(input: TensorHandle): TensorHandle {
    const kernel = this.toDepthwiseKernel(this.weights.depthwiseKernel);
    const output = this.backend.depthwiseConv1d(input, kernel, 1, 'same');
    if (kernel !== this.weights.depthwiseKernel) {
      this.backend.dispose(kernel);
    }
    return output;
  }

  private toDepthwiseKernel(kernel: TensorHandle): TensorHandle {
    const shape = this.backend.getShape(kernel);
    if (shape.length !== 3) {
      return kernel;
    }
    if (shape[1] === 1) {
      // PyTorch depthwise Conv1d: [C, 1, K] -> TF.js: [K, C, 1].
      return this.backend.transpose(kernel, [2, 0, 1]);
    }
    return kernel;
  }

  private extractConvState(input: TensorHandle): TensorHandle {
    const kernelShape = this.backend.getShape(this.weights.depthwiseKernel);
    const kernelSize = kernelShape.length === 3 ? kernelShape[kernelShape.length - 1] : 1;
    const keep = Math.max(1, kernelSize - 1);
    const shape = this.backend.getShape(input);
    const keepLen = Math.min(keep, shape[1]);
    const start = shape[1] - keepLen;
    return this.backend.slice(input, [0, start, 0], [shape[0], keepLen, shape[2]]);
  }

  private applyBatchNormOrAffine(x: TensorHandle): TensorHandle {
    if (this.weights.batchNorm.mean && this.weights.batchNorm.variance) {
      return this.backend.batchNorm(
        x,
        this.weights.batchNorm.mean,
        this.weights.batchNorm.variance,
        this.weights.batchNorm.scale,
        this.weights.batchNorm.offset,
        1e-5,
      );
    }

    // Some NeMo checkpoints only store affine BN parameters (weight/bias) in state_dict.
    const channels = this.backend.getShape(this.weights.batchNorm.scale)[0];
    const scale = this.backend.reshape(this.weights.batchNorm.scale, [1, 1, channels]);
    const offset = this.backend.reshape(this.weights.batchNorm.offset, [1, 1, channels]);
    const scaled = this.backend.mul(x, scale);
    const shifted = this.backend.add(scaled, offset);
    this.backend.dispose(scale);
    this.backend.dispose(offset);
    this.backend.dispose(scaled);
    return shifted;
  }
}
