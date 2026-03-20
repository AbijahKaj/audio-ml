import type { Shape, TensorHandle } from './types';

export interface ComputeBackend {
  tensor(data: Float32Array | Int32Array, shape: Shape, dtype?: 'float32' | 'int32'): TensorHandle;
  zeros(shape: Shape): TensorHandle;
  dispose(t: TensorHandle): void;

  matmul(a: TensorHandle, b: TensorHandle): TensorHandle;
  add(a: TensorHandle, b: TensorHandle): TensorHandle;
  sub(a: TensorHandle, b: TensorHandle): TensorHandle;
  mul(a: TensorHandle, b: TensorHandle): TensorHandle;
  div(a: TensorHandle, b: TensorHandle): TensorHandle;
  scale(a: TensorHandle, s: number): TensorHandle;

  softmax(x: TensorHandle, axis: number): TensorHandle;
  layerNorm(x: TensorHandle, weight: TensorHandle, bias: TensorHandle, eps: number): TensorHandle;
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

  conv2d(
    input: TensorHandle,
    filter: TensorHandle,
    strides: [number, number],
    padding: 'valid' | 'same',
  ): TensorHandle;
  depthwiseConv2d(
    input: TensorHandle,
    filter: TensorHandle,
    stride: [number, number],
    padding: 'valid' | 'same',
  ): TensorHandle;

  reshape(x: TensorHandle, shape: Shape): TensorHandle;
  transpose(x: TensorHandle, perm: number[]): TensorHandle;
  squeeze(x: TensorHandle, dims?: number[]): TensorHandle;
  expandDims(x: TensorHandle, axis: number): TensorHandle;
  slice(x: TensorHandle, begin: number[], size: number[]): TensorHandle;
  concat(tensors: TensorHandle[], axis: number): TensorHandle;
  stack(tensors: TensorHandle[], axis: number): TensorHandle;
  split(x: TensorHandle, numSplits: number, axis: number): TensorHandle[];
  gather(params: TensorHandle, indices: TensorHandle, axis: number, batchDims?: number): TensorHandle;
  pad(
    x: TensorHandle,
    paddings: Array<[number, number]>,
    constantValue?: number,
  ): TensorHandle;
  tile(x: TensorHandle, reps: number[]): TensorHandle;
  clipByValue(x: TensorHandle, clipMin: number, clipMax: number): TensorHandle;
  sum(x: TensorHandle, axis: number, keepDims?: boolean): TensorHandle;
  mean(x: TensorHandle, axis: number, keepDims?: boolean): TensorHandle;
  square(x: TensorHandle): TensorHandle;
  sqrt(x: TensorHandle): TensorHandle;
  neg(x: TensorHandle): TensorHandle;
  minimum(a: TensorHandle, b: TensorHandle): TensorHandle;
  maximum(a: TensorHandle, b: TensorHandle): TensorHandle;
  less(a: TensorHandle, b: TensorHandle | number): TensorHandle;
  where(condition: TensorHandle, a: TensorHandle, b: TensorHandle): TensorHandle;
  fill(shape: Shape, value: number): TensorHandle;
  cast(x: TensorHandle, dtype: 'float32' | 'int32'): TensorHandle;
  oneHot(indices: TensorHandle, depth: number, onValue?: number, offValue?: number): TensorHandle;

  getData(t: TensorHandle): Promise<Float32Array | Int32Array>;
  getShape(t: TensorHandle): Shape;
}
