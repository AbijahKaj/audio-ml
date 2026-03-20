import { ComputeScope } from '../compute/ComputeScope';
import type { ComputeBackend } from '../compute/Backend';
import type { TensorHandle } from '../compute/types';
import type { ConvSubsamplingWeights } from '../model/ModelWeights';
import { Linear } from './Linear';

export class ConvSubsampling {
  private outProjection: Linear;

  constructor(
    private backend: ComputeBackend,
    private weights: ConvSubsamplingWeights,
    private subsamplingFactor: number,
  ) {
    this.outProjection = new Linear(backend, weights.out);
  }

  forward(melFeatures: TensorHandle): TensorHandle {
    const scope = new ComputeScope();
    let x = scope.track(this.backend.expandDims(melFeatures, -1));
    x = scope.track(this.applyConv2d(x, this.weights.conv1.weight, this.weights.conv1.bias, [2, 2]));
    x = scope.track(this.backend.relu(x));

    x = scope.track(this.applyConv2d(x, this.weights.conv2.weight, this.weights.conv2.bias, [2, 2]));
    x = scope.track(this.backend.relu(x));

    if (this.weights.conv3) {
      x = scope.track(this.applyConv2d(x, this.weights.conv3.weight, this.weights.conv3.bias, [2, 1]));
      x = scope.track(this.backend.relu(x));
    }

    const shape = this.backend.getShape(x);
    const [batch, time, freq, channels] = shape;
    const flattened = scope.track(this.backend.reshape(x, [batch, time, freq * channels]));
    let projected = this.outProjection.forward(flattened);

    if (!this.weights.conv3 && this.subsamplingFactor >= 8) {
      projected = this.downsampleTime(projected, 2);
    }

    scope.keep(projected);
    scope.dispose(this.backend);
    return projected;
  }

  private applyConv2d(
    input: TensorHandle,
    kernel: TensorHandle,
    bias: TensorHandle | undefined,
    strides: [number, number],
  ): TensorHandle {
    const convertedKernel = this.toTfConv2DKernel(kernel);
    let out = this.backend.conv2d(input, convertedKernel, strides, 'same');
    if (convertedKernel !== kernel) {
      this.backend.dispose(convertedKernel);
    }
    if (bias) {
      const reshapedBias = this.backend.reshape(bias, [1, 1, 1, this.backend.getShape(bias)[0]]);
      const added = this.backend.add(out, reshapedBias);
      this.backend.dispose(reshapedBias);
      this.backend.dispose(out);
      out = added;
    }
    return out;
  }

  private toTfConv2DKernel(kernel: TensorHandle): TensorHandle {
    const shape = this.backend.getShape(kernel);
    if (shape.length !== 4) {
      return kernel;
    }

    // PyTorch format [out, in, kH, kW] -> TF.js [kH, kW, in, out].
    return this.backend.transpose(kernel, [2, 3, 1, 0]);
  }

  private downsampleTime(x: TensorHandle, stride: number): TensorHandle {
    const shape = this.backend.getShape(x);
    const time = shape[1];
    const indices = new Int32Array(Math.ceil(time / stride));
    for (let i = 0; i < indices.length; i++) {
      indices[i] = Math.min(time - 1, i * stride);
    }
    const indexTensor = this.backend.tensor(indices, [indices.length], 'int32');
    const output = this.backend.gather(x, indexTensor, 1);
    this.backend.dispose(indexTensor);
    return output;
  }
}
