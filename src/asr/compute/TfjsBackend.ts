import * as tf from '@tensorflow/tfjs';
import type { ComputeBackend } from './Backend';
import type { TensorHandle, Shape } from './types';

function T(h: TensorHandle): tf.Tensor {
  return h as tf.Tensor;
}

export class TfjsBackend implements ComputeBackend {
  async init(backend: 'wasm' | 'webgpu' | 'webgl' | 'cpu' = 'cpu'): Promise<void> {
    await tf.setBackend(backend);
    await tf.ready();
  }

  tensor(data: Float32Array | Int32Array, shape: Shape): TensorHandle {
    return tf.tensor(data, shape as number[]);
  }

  zeros(shape: Shape): TensorHandle {
    return tf.zeros(shape as number[]);
  }

  ones(shape: Shape): TensorHandle {
    return tf.ones(shape as number[]);
  }

  dispose(t: TensorHandle): void {
    T(t).dispose();
  }

  matmul(a: TensorHandle, b: TensorHandle): TensorHandle {
    return tf.matMul(T(a), T(b));
  }

  add(a: TensorHandle, b: TensorHandle): TensorHandle {
    return tf.add(T(a), T(b));
  }

  sub(a: TensorHandle, b: TensorHandle): TensorHandle {
    return tf.sub(T(a), T(b));
  }

  mul(a: TensorHandle, b: TensorHandle): TensorHandle {
    return tf.mul(T(a), T(b));
  }

  div(a: TensorHandle, b: TensorHandle): TensorHandle {
    return tf.div(T(a), T(b));
  }

  scale(a: TensorHandle, s: number): TensorHandle {
    return tf.mul(T(a), tf.scalar(s));
  }

  neg(a: TensorHandle): TensorHandle {
    return tf.neg(T(a));
  }

  sqrt(a: TensorHandle): TensorHandle {
    return tf.sqrt(T(a));
  }

  exp(a: TensorHandle): TensorHandle {
    return tf.exp(T(a));
  }

  log(a: TensorHandle): TensorHandle {
    return tf.log(T(a));
  }

  pow(a: TensorHandle, exponent: number): TensorHandle {
    return tf.pow(T(a), tf.scalar(exponent));
  }

  minimum(a: TensorHandle, b: TensorHandle): TensorHandle {
    return tf.minimum(T(a), T(b));
  }

  maximum(a: TensorHandle, b: TensorHandle): TensorHandle {
    return tf.maximum(T(a), T(b));
  }

  where(condition: TensorHandle, a: TensorHandle, b: TensorHandle): TensorHandle {
    return tf.where(T(condition) as tf.Tensor<tf.Rank>, T(a), T(b));
  }

  sum(x: TensorHandle, axis: number | number[], keepDims = false): TensorHandle {
    return tf.sum(T(x), axis, keepDims);
  }

  mean(x: TensorHandle, axis: number | number[], keepDims = false): TensorHandle {
    return tf.mean(T(x), axis, keepDims);
  }

  argmax(x: TensorHandle, axis: number): TensorHandle {
    return tf.argMax(T(x), axis);
  }

  softmax(x: TensorHandle, axis: number): TensorHandle {
    return tf.softmax(T(x), axis);
  }

  logSoftmax(x: TensorHandle, axis: number): TensorHandle {
    return tf.logSoftmax(T(x), axis);
  }

  layerNorm(x: TensorHandle, weight: TensorHandle, bias: TensorHandle, eps: number): TensorHandle {
    return tf.tidy(() => {
      const t = T(x);
      const axis = t.rank - 1;
      const { mean, variance } = tf.moments(t, [axis], true);
      const normalized = tf.div(tf.sub(t, mean), tf.sqrt(tf.add(variance, tf.scalar(eps))));
      return tf.add(tf.mul(normalized, T(weight)), T(bias));
    });
  }

  batchNorm(
    x: TensorHandle,
    mean: TensorHandle,
    variance: TensorHandle,
    scale: TensorHandle,
    offset: TensorHandle,
    eps: number
  ): TensorHandle {
    return tf.tidy(() => {
      const normalized = tf.div(tf.sub(T(x), T(mean)), tf.sqrt(tf.add(T(variance), tf.scalar(eps))));
      return tf.add(tf.mul(normalized, T(scale)), T(offset));
    });
  }

  relu(x: TensorHandle): TensorHandle {
    return tf.relu(T(x));
  }

  silu(x: TensorHandle): TensorHandle {
    return tf.tidy(() => {
      const t = T(x);
      return tf.mul(t, tf.sigmoid(t));
    });
  }

  gelu(x: TensorHandle): TensorHandle {
    return tf.tidy(() => {
      const t = T(x);
      const cdf = tf.mul(
        tf.scalar(0.5),
        tf.add(tf.scalar(1.0), tf.erf(tf.mul(t, tf.scalar(Math.SQRT1_2))))
      );
      return tf.mul(t, cdf);
    });
  }

  sigmoid(x: TensorHandle): TensorHandle {
    return tf.sigmoid(T(x));
  }

  tanh(x: TensorHandle): TensorHandle {
    return tf.tanh(T(x));
  }

  conv1d(
    input: TensorHandle,
    kernel: TensorHandle,
    stride: number,
    padding: number,
    bias?: TensorHandle
  ): TensorHandle {
    return tf.tidy(() => {
      let inp = T(input); // [B, T, C_in]
      if (padding > 0) {
        inp = tf.pad(inp, [[0, 0], [padding, padding], [0, 0]]) as tf.Tensor;
      }
      let result = tf.conv1d(inp as tf.Tensor3D, T(kernel) as tf.Tensor3D, stride, 'valid');
      if (bias) {
        result = tf.add(result, T(bias)) as tf.Tensor3D;
      }
      return result;
    });
  }

  conv2d(
    input: TensorHandle,
    kernel: TensorHandle,
    strides: [number, number],
    padding: 'valid' | 'same',
    bias?: TensorHandle
  ): TensorHandle {
    return tf.tidy(() => {
      let result = tf.conv2d(
        T(input) as tf.Tensor4D,
        T(kernel) as tf.Tensor4D,
        strides,
        padding
      );
      if (bias) {
        result = tf.add(result, T(bias)) as tf.Tensor4D;
      }
      return result;
    });
  }

  depthwiseConv1d(
    input: TensorHandle,
    kernel: TensorHandle,
    stride: number,
    padding: number
  ): TensorHandle {
    return tf.tidy(() => {
      let inp = T(input) as tf.Tensor3D; // [B, T, C]
      if (padding > 0) {
        inp = tf.pad(inp, [[0, 0], [padding, padding], [0, 0]]) as tf.Tensor3D;
      }
      const inp4d = inp.expandDims(1) as tf.Tensor4D; // [B, 1, T, C]
      const kern4d = T(kernel).expandDims(0) as tf.Tensor4D; // [1, K, C, 1]
      const result = tf.depthwiseConv2d(inp4d, kern4d, [1, stride], 'valid');
      return result.squeeze([1]);
    });
  }

  reshape(x: TensorHandle, shape: Shape): TensorHandle {
    return tf.reshape(T(x), shape as number[]);
  }

  transpose(x: TensorHandle, perm: number[]): TensorHandle {
    return tf.transpose(T(x), perm);
  }

  slice(x: TensorHandle, begin: number[], size: number[]): TensorHandle {
    return tf.slice(T(x), begin, size);
  }

  concat(tensors: TensorHandle[], axis: number): TensorHandle {
    return tf.concat(tensors.map(T), axis);
  }

  split(x: TensorHandle, numOrSizes: number | number[], axis: number): TensorHandle[] {
    return tf.split(T(x), numOrSizes, axis);
  }

  gather(x: TensorHandle, indices: TensorHandle, axis: number): TensorHandle {
    return tf.gather(T(x), T(indices) as tf.Tensor1D, axis);
  }

  expandDims(x: TensorHandle, axis: number): TensorHandle {
    return tf.expandDims(T(x), axis);
  }

  squeeze(x: TensorHandle, axis?: number[]): TensorHandle {
    return tf.squeeze(T(x), axis);
  }

  pad(x: TensorHandle, paddings: Array<[number, number]>, constantValue = 0): TensorHandle {
    return tf.pad(T(x), paddings, constantValue);
  }

  tile(x: TensorHandle, reps: number[]): TensorHandle {
    return tf.tile(T(x), reps);
  }

  broadcastTo(x: TensorHandle, shape: Shape): TensorHandle {
    return tf.broadcastTo(T(x), shape as number[]);
  }

  cast(x: TensorHandle, dtype: 'float32' | 'int32'): TensorHandle {
    return tf.cast(T(x), dtype);
  }

  scalarTensor(value: number): TensorHandle {
    return tf.scalar(value);
  }

  range(start: number, stop: number, step = 1): TensorHandle {
    return tf.range(start, stop, step);
  }

  clone(x: TensorHandle): TensorHandle {
    return tf.clone(T(x));
  }

  getData(t: TensorHandle): Promise<Float32Array> {
    return T(t).data() as Promise<Float32Array>;
  }

  getDataSync(t: TensorHandle): Float32Array {
    return T(t).dataSync() as Float32Array;
  }

  async getIntData(t: TensorHandle): Promise<Int32Array> {
    return T(t).data() as Promise<Int32Array>;
  }

  getShape(t: TensorHandle): Shape {
    return T(t).shape;
  }

  getSize(t: TensorHandle): number {
    return T(t).size;
  }

  tidy<R>(fn: () => R): R {
    return tf.tidy(fn as () => tf.TensorContainer) as R;
  }
}
