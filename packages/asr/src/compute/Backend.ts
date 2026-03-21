import type { TensorHandle, Shape } from './types';

export interface ComputeBackend {
  tensor(data: Float32Array | Int32Array, shape: Shape): TensorHandle;
  zeros(shape: Shape): TensorHandle;
  ones(shape: Shape): TensorHandle;
  dispose(t: TensorHandle): void;

  matmul(a: TensorHandle, b: TensorHandle): TensorHandle;
  add(a: TensorHandle, b: TensorHandle): TensorHandle;
  sub(a: TensorHandle, b: TensorHandle): TensorHandle;
  mul(a: TensorHandle, b: TensorHandle): TensorHandle;
  div(a: TensorHandle, b: TensorHandle): TensorHandle;
  scale(a: TensorHandle, s: number): TensorHandle;
  neg(a: TensorHandle): TensorHandle;
  sqrt(a: TensorHandle): TensorHandle;
  exp(a: TensorHandle): TensorHandle;
  log(a: TensorHandle): TensorHandle;
  pow(a: TensorHandle, exponent: number): TensorHandle;
  minimum(a: TensorHandle, b: TensorHandle): TensorHandle;
  maximum(a: TensorHandle, b: TensorHandle): TensorHandle;
  where(condition: TensorHandle, a: TensorHandle, b: TensorHandle): TensorHandle;

  sum(x: TensorHandle, axis: number | number[], keepDims?: boolean): TensorHandle;
  mean(x: TensorHandle, axis: number | number[], keepDims?: boolean): TensorHandle;
  argmax(x: TensorHandle, axis: number): TensorHandle;

  softmax(x: TensorHandle, axis: number): TensorHandle;
  logSoftmax(x: TensorHandle, axis: number): TensorHandle;
  layerNorm(x: TensorHandle, weight: TensorHandle, bias: TensorHandle, eps: number): TensorHandle;
  batchNorm(
    x: TensorHandle,
    mean: TensorHandle,
    variance: TensorHandle,
    scale: TensorHandle,
    offset: TensorHandle,
    eps: number
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
    padding: number,
    bias?: TensorHandle
  ): TensorHandle;
  conv2d(
    input: TensorHandle,
    kernel: TensorHandle,
    strides: [number, number],
    padding: 'valid' | 'same',
    bias?: TensorHandle
  ): TensorHandle;
  depthwiseConv1d(
    input: TensorHandle,
    kernel: TensorHandle,
    stride: number,
    padding: number
  ): TensorHandle;
  depthwiseConv2d(
    input: TensorHandle,
    kernel: TensorHandle,
    strides: [number, number],
    padding: 'valid' | 'same',
    bias?: TensorHandle
  ): TensorHandle;

  reshape(x: TensorHandle, shape: Shape): TensorHandle;
  transpose(x: TensorHandle, perm: number[]): TensorHandle;
  slice(x: TensorHandle, begin: number[], size: number[]): TensorHandle;
  concat(tensors: TensorHandle[], axis: number): TensorHandle;
  split(x: TensorHandle, numOrSizes: number | number[], axis: number): TensorHandle[];
  gather(x: TensorHandle, indices: TensorHandle, axis: number): TensorHandle;
  expandDims(x: TensorHandle, axis: number): TensorHandle;
  squeeze(x: TensorHandle, axis?: number[]): TensorHandle;
  pad(x: TensorHandle, paddings: Array<[number, number]>, constantValue?: number): TensorHandle;
  tile(x: TensorHandle, reps: number[]): TensorHandle;
  broadcastTo(x: TensorHandle, shape: Shape): TensorHandle;
  cast(x: TensorHandle, dtype: 'float32' | 'int32'): TensorHandle;
  scalarTensor(value: number): TensorHandle;
  range(start: number, stop: number, step?: number): TensorHandle;
  clone(x: TensorHandle): TensorHandle;

  getData(t: TensorHandle): Promise<Float32Array>;
  getDataSync(t: TensorHandle): Float32Array;
  getIntData(t: TensorHandle): Promise<Int32Array>;
  getShape(t: TensorHandle): Shape;
  getSize(t: TensorHandle): number;

  tidy<T>(fn: () => T): T;
}
