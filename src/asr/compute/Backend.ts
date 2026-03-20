import type { Dtype, Shape, TensorHandle } from './types';

/**
 * Pluggable compute backend so ASR code does not import TensorFlow.js directly.
 */
export interface ComputeBackend {
  tensor(data: Float32Array | Int32Array, shape: Shape, dtype?: Dtype): TensorHandle;
  zeros(shape: Shape): TensorHandle;
  ones(shape: Shape): TensorHandle;
  dispose(t: TensorHandle): void;

  matmul(a: TensorHandle, b: TensorHandle): TensorHandle;
  add(a: TensorHandle, b: TensorHandle): TensorHandle;
  sub(a: TensorHandle, b: TensorHandle): TensorHandle;
  mul(a: TensorHandle, b: TensorHandle): TensorHandle;
  div(a: TensorHandle, b: TensorHandle): TensorHandle;
  scale(a: TensorHandle, s: number): TensorHandle;

  softmax(x: TensorHandle, axis: number): TensorHandle;
  layerNorm(
    x: TensorHandle,
    weight: TensorHandle,
    bias: TensorHandle,
    eps: number,
  ): TensorHandle;
  batchNorm(
    x: TensorHandle,
    mean: TensorHandle,
    variance: TensorHandle,
    scale: TensorHandle,
    offset: TensorHandle,
    eps: number,
  ): TensorHandle;

  relu(x: TensorHandle): TensorHandle;
  silu(x: TensorHandle): TensorHandle;
  gelu(x: TensorHandle): TensorHandle;
  sigmoid(x: TensorHandle): TensorHandle;
  tanh(x: TensorHandle): TensorHandle;

  conv1d(
    input: TensorHandle,
    kernel: TensorHandle,
    stride: number,
    padding: 'valid' | 'same',
  ): TensorHandle;
  conv2d(
    input: TensorHandle,
    kernel: TensorHandle,
    strides: [number, number],
    padding: 'valid' | 'same',
  ): TensorHandle;
  depthwiseConv1d(
    input: TensorHandle,
    kernel: TensorHandle,
    stride: number,
    padding: 'valid' | 'same',
  ): TensorHandle;

  reshape(x: TensorHandle, shape: Shape): TensorHandle;
  transpose(x: TensorHandle, perm: number[]): TensorHandle;
  slice(x: TensorHandle, begin: number[], size: number[]): TensorHandle;
  concat(tensors: TensorHandle[], axis: number): TensorHandle;
  split(x: TensorHandle, numSplits: number, axis: number): TensorHandle[];
  gather(x: TensorHandle, indices: TensorHandle, axis: number, batchDims?: number): TensorHandle;
  squeeze(x: TensorHandle, axes?: number[]): TensorHandle;
  expandDims(x: TensorHandle, axis: number): TensorHandle;
  pad(
    x: TensorHandle,
    paddings: Array<[number, number]>,
    constantValue?: number,
  ): TensorHandle;

  getData(t: TensorHandle): Promise<Float32Array>;
  getShape(t: TensorHandle): Shape;
}