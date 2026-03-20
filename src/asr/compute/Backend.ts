import type { Dtype, Shape, TensorHandle } from './types';

export interface ComputeBackend {
  ready(): Promise<void>;

  // Tensor lifecycle.
  tensor(
    data: Float32Array | Int32Array | Int8Array | Uint8Array | number[],
    shape: Shape,
    dtype?: Dtype,
  ): TensorHandle;
  zeros(shape: Shape, dtype?: Dtype): TensorHandle;
  ones(shape: Shape, dtype?: Dtype): TensorHandle;
  dispose(tensor: TensorHandle): void;

  // Core math.
  matmul(a: TensorHandle, b: TensorHandle): TensorHandle;
  add(a: TensorHandle, b: TensorHandle): TensorHandle;
  sub(a: TensorHandle, b: TensorHandle): TensorHandle;
  mul(a: TensorHandle, b: TensorHandle): TensorHandle;
  div(a: TensorHandle, b: TensorHandle): TensorHandle;
  scale(a: TensorHandle, scalar: number): TensorHandle;

  // Normalization and reduction.
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

  // Activations.
  relu(x: TensorHandle): TensorHandle;
  silu(x: TensorHandle): TensorHandle;
  gelu(x: TensorHandle): TensorHandle;
  sigmoid(x: TensorHandle): TensorHandle;
  tanh(x: TensorHandle): TensorHandle;

  // Convolutions.
  conv1d(
    input: TensorHandle,
    kernel: TensorHandle,
    stride: number,
    padding: 'same' | 'valid',
  ): TensorHandle;
  conv2d(
    input: TensorHandle,
    kernel: TensorHandle,
    strides: [number, number],
    padding: 'same' | 'valid',
  ): TensorHandle;
  depthwiseConv1d(
    input: TensorHandle,
    kernel: TensorHandle,
    stride: number,
    padding: 'same' | 'valid',
  ): TensorHandle;

  // Shape and indexing.
  reshape(x: TensorHandle, shape: Shape): TensorHandle;
  transpose(x: TensorHandle, perm: number[]): TensorHandle;
  slice(x: TensorHandle, begin: number[], size: number[]): TensorHandle;
  concat(tensors: TensorHandle[], axis: number): TensorHandle;
  split(x: TensorHandle, numOrSizeSplits: number | number[], axis: number): TensorHandle[];
  gather(x: TensorHandle, indices: TensorHandle, axis: number): TensorHandle;
  squeeze(x: TensorHandle, axis?: number[]): TensorHandle;
  expandDims(x: TensorHandle, axis: number): TensorHandle;

  // Host transfer.
  getData(tensor: TensorHandle): Promise<Float32Array>;
  getIntData(tensor: TensorHandle): Promise<Int32Array>;
  getShape(tensor: TensorHandle): Shape;
}
