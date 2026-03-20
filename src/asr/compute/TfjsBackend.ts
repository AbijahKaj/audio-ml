import * as tf from '@tensorflow/tfjs';
import type { ComputeBackend } from './Backend';
import type { Dtype, Shape, TensorHandle } from './types';

export type TfjsBackendName = 'wasm' | 'webgpu' | 'webgl' | 'cpu';

/**
 * TensorFlow.js implementation of {@link ComputeBackend}.
 * Call {@link TfjsBackend.init} once before use (sets backend + warms up WASM).
 */
export class TfjsBackend implements ComputeBackend {
  static async init(preferred: TfjsBackendName = 'wasm'): Promise<TfjsBackendName> {
    await import('@tensorflow/tfjs-backend-wasm');
    const order: TfjsBackendName[] = [preferred, 'cpu', 'webgl'];
    for (const b of order) {
      try {
        await tf.setBackend(b);
        await tf.ready();
        return b;
      } catch {
        /* try next */
      }
    }
    await tf.ready();
    return tf.getBackend() as TfjsBackendName;
  }

  tensor(data: Float32Array | Int32Array, shape: Shape, dtype: Dtype = 'float32'): TensorHandle {
    if (dtype === 'int32') {
      return tf.tensor(data as Int32Array, [...shape], 'int32');
    }
    return tf.tensor(data, [...shape], 'float32');
  }

  zeros(shape: Shape): TensorHandle {
    return tf.zeros([...shape]);
  }

  ones(shape: Shape): TensorHandle {
    return tf.ones([...shape]);
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
    return tf.mul(a as tf.Tensor, tf.scalar(s));
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
      const ax = t.rank - 1;
      const moments = tf.moments(t, ax, true);
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
      const inner = tf.add(t, tf.mul(tf.scalar(0.044715), tf.pow(t, tf.scalar(3))));
      const tanhPart = tf.tanh(tf.mul(tf.scalar(Math.sqrt(2 / Math.PI)), inner));
      return tf.mul(tf.mul(tf.scalar(0.5), t), tf.add(tf.scalar(1), tanhPart));
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
    padding: 'valid' | 'same',
  ): TensorHandle {
    return tf.conv1d(input as tf.Tensor3D, kernel as tf.Tensor3D, stride, padding);
  }

  conv2d(
    input: TensorHandle,
    kernel: TensorHandle,
    strides: [number, number],
    padding: 'valid' | 'same',
  ): TensorHandle {
    return tf.conv2d(input as tf.Tensor4D, kernel as tf.Tensor4D, strides, padding);
  }

  depthwiseConv1d(
    input: TensorHandle,
    kernel: TensorHandle,
    stride: number,
    padding: 'valid' | 'same',
  ): TensorHandle {
    return tf.tidy(() => {
      const inp = input as tf.Tensor;
      const inp4 = tf.expandDims(inp, 1);
      const pad: 'same' | 'valid' = padding;
      const out4 = tf.depthwiseConv2d(
        inp4 as tf.Tensor4D,
        kernel as tf.Tensor4D,
        [1, stride],
        pad,
      );
      return tf.squeeze(out4, [1]);
    });
  }

  reshape(x: TensorHandle, shape: Shape): TensorHandle {
    return tf.reshape(x as tf.Tensor, [...shape]);
  }

  transpose(x: TensorHandle, perm: number[]): TensorHandle {
    return tf.transpose(x as tf.Tensor, perm);
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

  split(x: TensorHandle, numSplits: number, axis: number): TensorHandle[] {
    return tf.split(x as tf.Tensor, numSplits, axis) as unknown as TensorHandle[];
  }

  gather(x: TensorHandle, indices: TensorHandle, axis: number, batchDims = 0): TensorHandle {
    return tf.gather(x as tf.Tensor, indices as tf.Tensor, axis, batchDims);
  }

  squeeze(x: TensorHandle, axes?: number[]): TensorHandle {
    return axes ? tf.squeeze(x as tf.Tensor, axes) : tf.squeeze(x as tf.Tensor);
  }

  expandDims(x: TensorHandle, axis: number): TensorHandle {
    return tf.expandDims(x as tf.Tensor, axis);
  }

  pad(
    x: TensorHandle,
    paddings: Array<[number, number]>,
    constantValue = 0,
  ): TensorHandle {
    return tf.pad(x as tf.Tensor, paddings, constantValue);
  }

  async getData(t: TensorHandle): Promise<Float32Array> {
    const d = await (t as tf.Tensor).data();
    return d instanceof Float32Array ? d : Float32Array.from(d);
  }

  getShape(t: TensorHandle): Shape {
    return (t as tf.Tensor).shape;
  }
}