import * as tf from '@tensorflow/tfjs';
import type { ComputeBackend } from './Backend';
import type { Shape, TensorHandle } from './types';

export type TfjsBackendName = 'cpu' | 'webgl';

const INF_MASK = -1e9;

export class TfjsBackend implements ComputeBackend {
  constructor(backend: TfjsBackendName = 'cpu') {
    void tf.setBackend(backend);
  }

  /**
   * WASM must be loaded before calling (registers the backend with tfjs).
   * @example await import('@tensorflow/tfjs-backend-wasm'); await TfjsBackend.useWasmBackend();
   */
  static async useWasmBackend(): Promise<void> {
    await import('@tensorflow/tfjs-backend-wasm');
    await tf.setBackend('wasm');
    await tf.ready();
  }

  tensor(data: Float32Array | Int32Array, shape: Shape, dtype: 'float32' | 'int32' = 'float32'): TensorHandle {
    if (dtype === 'int32') {
      return tf.tensor(Array.from(data), shape as number[], 'int32');
    }
    return tf.tensor(data, shape as number[], 'float32');
  }

  zeros(shape: Shape): TensorHandle {
    return tf.zeros(shape as number[]);
  }

  dispose(t: TensorHandle): void {
    (t as tf.Tensor).dispose();
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

  scale(a: TensorHandle, s: number): TensorHandle {
    return tf.mul(a as tf.Tensor, s);
  }

  softmax(x: TensorHandle, axis: number): TensorHandle {
    return tf.softmax(x as tf.Tensor, axis);
  }

  layerNorm(x: TensorHandle, weight: TensorHandle, bias: TensorHandle, eps: number): TensorHandle {
    return tf.tidy(() => {
      const t = x as tf.Tensor;
      const lastAxis = t.rank - 1;
      const moments = tf.moments(t, lastAxis, true);
      const normalized = tf.div(tf.sub(t, moments.mean), tf.sqrt(tf.add(moments.variance, eps)));
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
      const m = mean as tf.Tensor;
      const v = variance as tf.Tensor;
      const g = scale as tf.Tensor;
      const b = offset as tf.Tensor;
      const norm = tf.div(tf.sub(t, m), tf.sqrt(tf.add(v, eps)));
      return tf.add(tf.mul(norm, g), b);
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
      const t = x as tf.Tensor;
      const c = Math.sqrt(2 / Math.PI);
      const t3 = tf.mul(t, tf.mul(t, t));
      const inner = tf.add(t, tf.mul(0.044715, t3));
      return tf.mul(0.5, tf.mul(t, tf.add(1, tf.tanh(tf.mul(c, inner)))));
    });
  }

  sigmoid(x: TensorHandle): TensorHandle {
    return tf.sigmoid(x as tf.Tensor);
  }

  tanh(x: TensorHandle): TensorHandle {
    return tf.tanh(x as tf.Tensor);
  }

  conv2d(
    input: TensorHandle,
    filter: TensorHandle,
    strides: [number, number],
    padding: 'valid' | 'same',
  ): TensorHandle {
    return tf.conv2d(
      input as tf.Tensor4D,
      filter as tf.Tensor4D,
      strides,
      padding,
    );
  }

  depthwiseConv2d(
    input: TensorHandle,
    filter: TensorHandle,
    stride: [number, number],
    padding: 'valid' | 'same',
  ): TensorHandle {
    return tf.depthwiseConv2d(
      input as tf.Tensor4D,
      filter as tf.Tensor4D,
      stride,
      padding,
    );
  }

  reshape(x: TensorHandle, shape: Shape): TensorHandle {
    return tf.reshape(x as tf.Tensor, shape as number[]);
  }

  transpose(x: TensorHandle, perm: number[]): TensorHandle {
    return tf.transpose(x as tf.Tensor, perm);
  }

  squeeze(x: TensorHandle, dims?: number[]): TensorHandle {
    return tf.squeeze(x as tf.Tensor3D | tf.Tensor4D, dims);
  }

  expandDims(x: TensorHandle, axis: number): TensorHandle {
    return tf.expandDims(x as tf.Tensor, axis);
  }

  slice(x: TensorHandle, begin: number[], size: number[]): TensorHandle {
    return tf.slice(x as tf.Tensor, begin, size);
  }

  concat(tensors: TensorHandle[], axis: number): TensorHandle {
    return tf.concat(
      tensors.map(t => t as tf.Tensor),
      axis,
    );
  }

  stack(tensors: TensorHandle[], axis: number): TensorHandle {
    return tf.stack(
      tensors.map(t => t as tf.Tensor),
      axis,
    );
  }

  split(x: TensorHandle, numSplits: number, axis: number): TensorHandle[] {
    return tf.split(x as tf.Tensor, numSplits, axis) as unknown as TensorHandle[];
  }

  gather(params: TensorHandle, indices: TensorHandle, axis: number, batchDims = 0): TensorHandle {
    return tf.gather(params as tf.Tensor, indices as tf.Tensor, axis, batchDims);
  }

  pad(x: TensorHandle, paddings: Array<[number, number]>, constantValue = 0): TensorHandle {
    return tf.pad(x as tf.Tensor, paddings, constantValue);
  }

  tile(x: TensorHandle, reps: number[]): TensorHandle {
    return tf.tile(x as tf.Tensor, reps);
  }

  clipByValue(x: TensorHandle, clipMin: number, clipMax: number): TensorHandle {
    return tf.clipByValue(x as tf.Tensor, clipMin, clipMax);
  }

  sum(x: TensorHandle, axis: number, keepDims = false): TensorHandle {
    return tf.sum(x as tf.Tensor, axis, keepDims);
  }

  mean(x: TensorHandle, axis: number, keepDims = false): TensorHandle {
    return tf.mean(x as tf.Tensor, axis, keepDims);
  }

  square(x: TensorHandle): TensorHandle {
    return tf.square(x as tf.Tensor);
  }

  sqrt(x: TensorHandle): TensorHandle {
    return tf.sqrt(x as tf.Tensor);
  }

  neg(x: TensorHandle): TensorHandle {
    return tf.neg(x as tf.Tensor);
  }

  minimum(a: TensorHandle, b: TensorHandle): TensorHandle {
    return tf.minimum(a as tf.Tensor, b as tf.Tensor);
  }

  maximum(a: TensorHandle, b: TensorHandle): TensorHandle {
    return tf.maximum(a as tf.Tensor, b as tf.Tensor);
  }

  less(a: TensorHandle, b: TensorHandle | number): TensorHandle {
    const bt = typeof b === 'number' ? tf.scalar(b) : (b as tf.Tensor);
    return tf.less(a as tf.Tensor, bt);
  }

  where(condition: TensorHandle, a: TensorHandle, b: TensorHandle): TensorHandle {
    return tf.where(condition as tf.Tensor, a as tf.Tensor, b as tf.Tensor);
  }

  fill(shape: Shape, value: number): TensorHandle {
    return tf.fill(shape as number[], value);
  }

  cast(x: TensorHandle, dtype: 'float32' | 'int32'): TensorHandle {
    return tf.cast(x as tf.Tensor, dtype);
  }

  oneHot(indices: TensorHandle, depth: number, onValue = 1, offValue = 0): TensorHandle {
    return tf.oneHot(indices as tf.Tensor, depth, onValue, offValue);
  }

  async getData(t: TensorHandle): Promise<Float32Array | Int32Array> {
    const tensor = t as tf.Tensor;
    const arr = await tensor.data();
    if (tensor.dtype === 'int32') {
      return arr as Int32Array;
    }
    return arr as Float32Array;
  }

  getShape(t: TensorHandle): Shape {
    return (t as tf.Tensor).shape;
  }

  /** Attention mask fill value (NeMo uses large negative). */
  static attentionMaskValue(): number {
    return INF_MASK;
  }
}
