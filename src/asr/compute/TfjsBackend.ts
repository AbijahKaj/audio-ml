import * as tf from '@tensorflow/tfjs';
import type { ComputeBackend, TensorHandle, Shape } from './index.js';

/**
 * TensorFlow.js implementation of the ComputeBackend.
 * Wraps tf.* ops. All model code calls this interface — never tf.* directly.
 */
export class TfjsBackend implements ComputeBackend {
  constructor(backend: 'wasm' | 'webgpu' | 'webgl' | 'cpu' = 'cpu') {
    tf.setBackend(backend);
  }

  tensor(data: Float32Array | Int32Array, shape: Shape): TensorHandle {
    if (data instanceof Int32Array) {
      return tf.tensor(Array.from(data), shape as number[], 'int32');
    }
    return tf.tensor(data, shape as number[], 'float32');
  }

  zeros(shape: Shape): TensorHandle {
    return tf.zeros(shape as number[]);
  }

  ones(shape: Shape): TensorHandle {
    return tf.ones(shape as number[]);
  }

  scalar(value: number): TensorHandle {
    return tf.scalar(value);
  }

  dispose(t: TensorHandle): void {
    if (t && typeof t.dispose === 'function') {
      (t as tf.Tensor).dispose();
    }
  }

  matmul(a: TensorHandle, b: TensorHandle, transposeA = false, transposeB = false): TensorHandle {
    return tf.matMul(a as tf.Tensor2D, b as tf.Tensor2D, transposeA, transposeB);
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

  scale(a: TensorHandle, s: number): TensorHandle {
    return tf.mul(a as tf.Tensor, s);
  }

  sqrt(x: TensorHandle): TensorHandle {
    return tf.sqrt(x as tf.Tensor);
  }

  exp(x: TensorHandle): TensorHandle {
    return tf.exp(x as tf.Tensor);
  }

  log(x: TensorHandle): TensorHandle {
    return tf.log(x as tf.Tensor);
  }

  pow(x: TensorHandle, exponent: number): TensorHandle {
    return tf.pow(x as tf.Tensor, exponent);
  }

  sum(x: TensorHandle, axis: number, keepDims = false): TensorHandle {
    return tf.sum(x as tf.Tensor, axis, keepDims);
  }

  mean(x: TensorHandle, axis: number, keepDims = false): TensorHandle {
    return tf.mean(x as tf.Tensor, axis, keepDims);
  }

  max(x: TensorHandle, axis: number, keepDims = false): TensorHandle {
    return tf.max(x as tf.Tensor, axis, keepDims);
  }

  softmax(x: TensorHandle, axis = -1): TensorHandle {
    return tf.softmax(x as tf.Tensor, axis);
  }

  argmax(x: TensorHandle, axis: number): TensorHandle {
    return tf.argMax(x as tf.Tensor, axis);
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
    return tf.tidy(() => {
      const t = x as tf.Tensor;
      const normalized = tf.div(
        tf.sub(t, mean as tf.Tensor),
        tf.sqrt(tf.add(variance as tf.Tensor, eps)),
      );
      return tf.add(tf.mul(normalized, scale as tf.Tensor), offset as tf.Tensor);
    });
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
      // GELU(x) = x * Φ(x), approximated via tanh
      const t = x as tf.Tensor;
      const c = 0.044715;
      const inner = tf.mul(
        Math.sqrt(2 / Math.PI),
        tf.add(t, tf.mul(c, tf.pow(t, 3))),
      );
      return tf.mul(tf.mul(0.5, t), tf.add(1, tf.tanh(inner)));
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
    bias: TensorHandle | null,
    stride: number,
    padding: 'same' | 'valid',
  ): TensorHandle {
    return tf.tidy(() => {
      // input: [B, T, C_in], kernel: [K, C_in, C_out]
      const result = tf.conv1d(
        input as tf.Tensor3D,
        kernel as tf.Tensor3D,
        stride,
        padding,
      );
      if (bias) return tf.add(result, bias as tf.Tensor);
      return result;
    });
  }

  conv2d(
    input: TensorHandle,
    kernel: TensorHandle,
    bias: TensorHandle | null,
    strides: [number, number],
    padding: 'same' | 'valid',
  ): TensorHandle {
    return tf.tidy(() => {
      // input: [B, H, W, C_in], kernel: [kH, kW, C_in, C_out]
      const result = tf.conv2d(
        input as tf.Tensor4D,
        kernel as tf.Tensor4D,
        strides,
        padding,
      );
      if (bias) return tf.add(result, bias as tf.Tensor);
      return result;
    });
  }

  depthwiseConv1d(
    input: TensorHandle,
    kernel: TensorHandle,
    stride: number,
    padding: 'same' | 'valid',
  ): TensorHandle {
    return tf.tidy(() => {
      // tfjs only has depthwiseConv2d, so we promote to 4D
      // input: [B, T, C] → [B, 1, T, C]
      // kernel: [K, C] → [1, K, C, 1]
      const inp4d = (input as tf.Tensor3D).expandDims(1) as tf.Tensor4D;
      const kern = (kernel as tf.Tensor2D).expandDims(0).expandDims(3) as tf.Tensor4D;
      const result = tf.depthwiseConv2d(inp4d, kern, [1, stride], padding);
      return result.squeeze([1]) as tf.Tensor3D;
    });
  }

  reshape(x: TensorHandle, shape: Shape): TensorHandle {
    return tf.reshape(x as tf.Tensor, shape as number[]);
  }

  transpose(x: TensorHandle, perm: number[]): TensorHandle {
    return tf.transpose(x as tf.Tensor, perm);
  }

  expandDims(x: TensorHandle, axis: number): TensorHandle {
    return tf.expandDims(x as tf.Tensor, axis);
  }

  squeeze(x: TensorHandle, axis?: number[]): TensorHandle {
    return tf.squeeze(x as tf.Tensor, axis);
  }

  slice(x: TensorHandle, begin: number[], size: number[]): TensorHandle {
    return tf.slice(x as tf.Tensor, begin, size);
  }

  concat(tensors: TensorHandle[], axis: number): TensorHandle {
    return tf.concat(tensors as tf.Tensor[], axis);
  }

  split(x: TensorHandle, numOrSizeSplits: number | number[], axis: number): TensorHandle[] {
    return tf.split(x as tf.Tensor, numOrSizeSplits, axis) as TensorHandle[];
  }

  gather(x: TensorHandle, indices: TensorHandle, axis = 0): TensorHandle {
    return tf.gather(x as tf.Tensor, indices as tf.Tensor1D, axis);
  }

  pad(x: TensorHandle, paddings: Array<[number, number]>): TensorHandle {
    return tf.pad(x as tf.Tensor, paddings);
  }

  tile(x: TensorHandle, reps: number[]): TensorHandle {
    return tf.tile(x as tf.Tensor, reps);
  }

  stack(tensors: TensorHandle[], axis = 0): TensorHandle {
    return tf.stack(tensors as tf.Tensor[], axis);
  }

  unstack(x: TensorHandle, axis = 0): TensorHandle[] {
    return tf.unstack(x as tf.Tensor, axis) as TensorHandle[];
  }

  fill(shape: Shape, value: number): TensorHandle {
    return tf.fill(shape as number[], value);
  }

  where(condition: TensorHandle, x: TensorHandle, y: TensorHandle): TensorHandle {
    return tf.where(condition as tf.Tensor, x as tf.Tensor, y as tf.Tensor);
  }

  async getData(t: TensorHandle): Promise<Float32Array> {
    const data = await (t as tf.Tensor).data();
    return data as Float32Array;
  }

  async getInt32Data(t: TensorHandle): Promise<Int32Array> {
    const data = await (t as tf.Tensor).data();
    return data as Int32Array;
  }

  getShape(t: TensorHandle): Shape {
    return (t as tf.Tensor).shape;
  }
}
