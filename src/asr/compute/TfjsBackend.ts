import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-wasm';

import type { ComputeBackend } from './Backend';
import type { BackendKind, Dtype, Shape, TensorHandle } from './types';

function toTensor(tensor: TensorHandle): tf.Tensor {
  return tensor as tf.Tensor;
}

function toTfDtype(dtype: Dtype): tf.DataType {
  if (dtype === 'int32' || dtype === 'int64') {
    return 'int32';
  }
  return 'float32';
}

export class TfjsBackend implements ComputeBackend {
  readonly backendKind: BackendKind;
  private readonly initPromise: Promise<void>;

  constructor(backendKind: BackendKind = 'wasm') {
    this.backendKind = backendKind;
    this.initPromise = (async () => {
      await tf.setBackend(backendKind);
      await tf.ready();
    })();
  }

  async ready(): Promise<void> {
    await this.initPromise;
  }

  tensor(data: Float32Array | Int32Array | number[], shape: Shape, dtype: Dtype = 'float32'): TensorHandle {
    return tf.tensor(data, shape as number[], toTfDtype(dtype));
  }

  zeros(shape: Shape, dtype: Dtype = 'float32'): TensorHandle {
    return tf.zeros(shape as number[], toTfDtype(dtype));
  }

  zerosLike(tensor: TensorHandle): TensorHandle {
    return tf.zerosLike(toTensor(tensor));
  }

  dispose(tensor: TensorHandle): void {
    toTensor(tensor).dispose();
  }

  disposeMany(tensors: TensorHandle[]): void {
    for (const tensor of tensors) {
      this.dispose(tensor);
    }
  }

  matmul(a: TensorHandle, b: TensorHandle): TensorHandle {
    return tf.matMul(toTensor(a), toTensor(b));
  }

  add(a: TensorHandle, b: TensorHandle): TensorHandle {
    return tf.add(toTensor(a), toTensor(b));
  }

  sub(a: TensorHandle, b: TensorHandle): TensorHandle {
    return tf.sub(toTensor(a), toTensor(b));
  }

  mul(a: TensorHandle, b: TensorHandle): TensorHandle {
    return tf.mul(toTensor(a), toTensor(b));
  }

  div(a: TensorHandle, b: TensorHandle): TensorHandle {
    return tf.div(toTensor(a), toTensor(b));
  }

  scale(a: TensorHandle, scalar: number): TensorHandle {
    return tf.mul(toTensor(a), scalar);
  }

  softmax(x: TensorHandle, axis: number): TensorHandle {
    return tf.softmax(toTensor(x), axis);
  }

  layerNorm(x: TensorHandle, weight: TensorHandle, bias: TensorHandle, eps: number): TensorHandle {
    return tf.tidy(() => {
      const tensor = toTensor(x);
      const axis = tensor.rank - 1;
      const { mean, variance } = tf.moments(tensor, axis, true);
      const normalized = tf.div(tf.sub(tensor, mean), tf.sqrt(tf.add(variance, eps)));
      return tf.add(tf.mul(normalized, toTensor(weight)), toTensor(bias));
    });
  }

  batchNorm(
    x: TensorHandle,
    mean: TensorHandle,
    variance: TensorHandle,
    scale: TensorHandle,
    offset: TensorHandle,
    eps: number,
  ): TensorHandle {
    return tf.batchNorm(
      toTensor(x),
      toTensor(mean),
      toTensor(variance),
      toTensor(offset),
      toTensor(scale),
      eps,
    );
  }

  relu(x: TensorHandle): TensorHandle {
    return tf.relu(toTensor(x));
  }

  silu(x: TensorHandle): TensorHandle {
    return tf.tidy(() => {
      const tensor = toTensor(x);
      return tf.mul(tensor, tf.sigmoid(tensor));
    });
  }

  gelu(x: TensorHandle): TensorHandle {
    return tf.tidy(() => {
      const tensor = toTensor(x);
      return tf.mul(
        tensor,
        tf.mul(
          0.5,
          tf.add(
            1,
            tf.erf(tf.mul(tensor, 1 / Math.sqrt(2))),
          ),
        ),
      );
    });
  }

  sigmoid(x: TensorHandle): TensorHandle {
    return tf.sigmoid(toTensor(x));
  }

  tanh(x: TensorHandle): TensorHandle {
    return tf.tanh(toTensor(x));
  }

  conv1d(input: TensorHandle, kernel: TensorHandle, stride: number, padding: number): TensorHandle {
    return tf.conv1d(
      toTensor(input) as tf.Tensor3D,
      toTensor(kernel) as tf.Tensor3D,
      stride,
      padding > 0 ? 'same' : 'valid',
    );
  }

  conv2d(input: TensorHandle, kernel: TensorHandle, strides: [number, number], padding: 'same' | 'valid'): TensorHandle {
    return tf.conv2d(
      toTensor(input) as tf.Tensor4D,
      toTensor(kernel) as tf.Tensor4D,
      strides,
      padding,
    );
  }

  depthwiseConv1d(input: TensorHandle, kernel: TensorHandle, stride: number, padding: number): TensorHandle {
    return tf.tidy(() => {
      const inputTensor = toTensor(input) as tf.Tensor3D;
      const input4d = inputTensor.expandDims(1);
      const kernelTensor = toTensor(kernel);
      const filter = kernelTensor.rank === 3 ? kernelTensor.expandDims(0) : kernelTensor;
      const output = tf.depthwiseConv2d(
        input4d as tf.Tensor4D,
        filter as tf.Tensor4D,
        [1, stride],
        padding > 0 ? 'same' : 'valid',
      );
      return output.squeeze([1]);
    });
  }

  reshape(x: TensorHandle, shape: Shape): TensorHandle {
    return tf.reshape(toTensor(x), shape as number[]);
  }

  transpose(x: TensorHandle, perm: number[]): TensorHandle {
    return tf.transpose(toTensor(x), perm);
  }

  slice(x: TensorHandle, begin: number[], size: number[]): TensorHandle {
    return tf.slice(toTensor(x), begin, size);
  }

  concat(tensors: TensorHandle[], axis: number): TensorHandle {
    return tf.concat(tensors.map((tensor) => toTensor(tensor)), axis);
  }

  split(x: TensorHandle, numSplits: number, axis: number): TensorHandle[] {
    return tf.split(toTensor(x), numSplits, axis);
  }

  gather(x: TensorHandle, indices: TensorHandle, axis: number): TensorHandle {
    return tf.gather(toTensor(x), toTensor(indices), axis);
  }

  async getData(tensor: TensorHandle): Promise<Float32Array> {
    const data = await toTensor(tensor).data();
    return data instanceof Float32Array ? data : Float32Array.from(data);
  }

  getShape(tensor: TensorHandle): Shape {
    return toTensor(tensor).shape;
  }
}
