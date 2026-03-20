import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-wasm';
import type { ComputeBackend } from './Backend';
import type { Dtype, Shape, TensorHandle } from './types';

type BackendType = 'wasm' | 'webgpu' | 'webgl' | 'cpu';

function toTfDtype(dtype?: Dtype): tf.DataType {
  if (!dtype || dtype === 'float32') return 'float32';
  if (dtype === 'float16') return 'float32';
  if (dtype === 'int32' || dtype === 'int64') return 'int32';
  return 'bool';
}

export class TfjsBackend implements ComputeBackend {
  private backendReady: Promise<void>;

  constructor(private backendName: BackendType = 'wasm') {
    this.backendReady = this.initializeBackend();
  }

  async ready(): Promise<void> {
    await this.backendReady;
  }

  tensor(
    data: Float32Array | Int32Array | Int8Array | Uint8Array | number[],
    shape: Shape,
    dtype: Dtype = 'float32',
  ): TensorHandle {
    return tf.tensor(data, [...shape], toTfDtype(dtype));
  }

  zeros(shape: Shape, dtype: Dtype = 'float32'): TensorHandle {
    return tf.zeros([...shape], toTfDtype(dtype));
  }

  ones(shape: Shape, dtype: Dtype = 'float32'): TensorHandle {
    return tf.ones([...shape], toTfDtype(dtype));
  }

  dispose(tensor: TensorHandle): void {
    (tensor as tf.Tensor).dispose();
  }

  matmul(a: TensorHandle, b: TensorHandle): TensorHandle {
    return tf.matMul(a as tf.Tensor, b as tf.Tensor);
  }

  add(a: TensorHandle, b: TensorHandle): TensorHandle {
    return tf.add(a as tf.Tensor, b as tf.Tensor);
  }

  sub(a: TensorHandle, b: TensorHandle): TensorHandle {
    return tf.sub(a as tf.Tensor, b as tf.Tensor);
  }

  mul(a: TensorHandle, b: TensorHandle): TensorHandle {
    return tf.mul(a as tf.Tensor, b as tf.Tensor);
  }

  div(a: TensorHandle, b: TensorHandle): TensorHandle {
    return tf.div(a as tf.Tensor, b as tf.Tensor);
  }

  scale(a: TensorHandle, scalar: number): TensorHandle {
    return tf.mul(a as tf.Tensor, scalar);
  }

  softmax(x: TensorHandle, axis: number): TensorHandle {
    return tf.softmax(x as tf.Tensor, axis);
  }

  layerNorm(
    x: TensorHandle,
    weight: TensorHandle,
    bias: TensorHandle,
    eps: number,
  ): TensorHandle {
    return tf.tidy(() => {
      const t = x as tf.Tensor;
      const axis = t.rank - 1;
      const { mean, variance } = tf.moments(t, [axis], true);
      const normalized = tf.div(tf.sub(t, mean), tf.sqrt(tf.add(variance, eps)));
      return tf.add(tf.mul(normalized, weight as tf.Tensor), bias as tf.Tensor);
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
      x as tf.Tensor,
      mean as tf.Tensor,
      variance as tf.Tensor,
      offset as tf.Tensor,
      scale as tf.Tensor,
      eps,
    );
  }

  relu(x: TensorHandle): TensorHandle {
    return tf.relu(x as tf.Tensor);
  }

  silu(x: TensorHandle): TensorHandle {
    return tf.tidy(() => {
      const t = x as tf.Tensor;
      return tf.mul(t, tf.sigmoid(t));
    });
  }

  gelu(x: TensorHandle): TensorHandle {
    return tf.tidy(() => {
      const t = x as tf.Tensor;
      return tf.mul(
        t,
        tf.mul(
          0.5,
          tf.add(
            1,
            tf.tanh(
              tf.mul(
                0.7978845608028654,
                tf.add(t, tf.mul(0.044715, tf.pow(t, 3))),
              ),
            ),
          ),
        ),
      );
    });
  }

  sigmoid(x: TensorHandle): TensorHandle {
    return tf.sigmoid(x as tf.Tensor);
  }

  tanh(x: TensorHandle): TensorHandle {
    return tf.tanh(x as tf.Tensor);
  }

  conv1d(
    input: TensorHandle,
    kernel: TensorHandle,
    stride: number,
    padding: 'same' | 'valid',
  ): TensorHandle {
    return tf.conv1d(input as tf.Tensor3D, kernel as tf.Tensor3D, stride, padding);
  }

  conv2d(
    input: TensorHandle,
    kernel: TensorHandle,
    strides: [number, number],
    padding: 'same' | 'valid',
  ): TensorHandle {
    return tf.conv2d(input as tf.Tensor4D, kernel as tf.Tensor4D, strides, padding);
  }

  depthwiseConv1d(
    input: TensorHandle,
    kernel: TensorHandle,
    stride: number,
    padding: 'same' | 'valid',
  ): TensorHandle {
    return tf.tidy(() => {
      const input4d = (input as tf.Tensor3D).expandDims(1);
      const kernel4d = (kernel as tf.Tensor3D).expandDims(0);
      const output = tf.depthwiseConv2d(
        input4d as tf.Tensor4D,
        kernel4d as tf.Tensor4D,
        [1, stride],
        padding,
      );
      return output.squeeze([1]);
    });
  }

  reshape(x: TensorHandle, shape: Shape): TensorHandle {
    return tf.reshape(x as tf.Tensor, [...shape]);
  }

  transpose(x: TensorHandle, perm: number[]): TensorHandle {
    return tf.transpose(x as tf.Tensor, perm);
  }

  slice(x: TensorHandle, begin: number[], size: number[]): TensorHandle {
    const tensor = x as tf.Tensor;
    const normalized = size.map((value, index) => {
      if (value !== -1) return value;
      return tensor.shape[index] - begin[index];
    });
    return tf.slice(tensor, begin, normalized);
  }

  concat(tensors: TensorHandle[], axis: number): TensorHandle {
    return tf.concat(tensors as tf.Tensor[], axis);
  }

  split(x: TensorHandle, numOrSizeSplits: number | number[], axis: number): TensorHandle[] {
    return tf.split(x as tf.Tensor, numOrSizeSplits, axis);
  }

  gather(x: TensorHandle, indices: TensorHandle, axis: number): TensorHandle {
    return tf.gather(x as tf.Tensor, indices as tf.Tensor, axis);
  }

  squeeze(x: TensorHandle, axis?: number[]): TensorHandle {
    return tf.squeeze(x as tf.Tensor, axis);
  }

  expandDims(x: TensorHandle, axis: number): TensorHandle {
    return tf.expandDims(x as tf.Tensor, axis);
  }

  async getData(tensor: TensorHandle): Promise<Float32Array> {
    const castTensor = tf.cast(tensor as tf.Tensor, 'float32');
    const data = await castTensor.data();
    castTensor.dispose();
    return data instanceof Float32Array ? data : Float32Array.from(data);
  }

  async getIntData(tensor: TensorHandle): Promise<Int32Array> {
    const castTensor = tf.cast(tensor as tf.Tensor, 'int32');
    const data = await castTensor.data();
    castTensor.dispose();
    return data instanceof Int32Array ? data : Int32Array.from(data);
  }

  getShape(tensor: TensorHandle): Shape {
    return (tensor as tf.Tensor).shape;
  }

  private async initializeBackend(): Promise<void> {
    await tf.ready();
    const didSet = await tf.setBackend(this.backendName).catch(() => false);
    if (!didSet) {
      await tf.setBackend('cpu');
    }
    await tf.ready();
  }
}
