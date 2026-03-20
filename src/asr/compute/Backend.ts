import type { BackendKind, Dtype, Shape, TensorHandle } from './types.js';

export interface ComputeBackend {
  readonly backendKind: BackendKind;

  ready(): Promise<void>;
  tensor(data: Float32Array | Int32Array | number[], shape: Shape, dtype?: Dtype): TensorHandle;
  zeros(shape: Shape, dtype?: Dtype): TensorHandle;
  zerosLike(tensor: TensorHandle): TensorHandle;
  dispose(tensor: TensorHandle): void;
  disposeMany(tensors: TensorHandle[]): void;

  matmul(a: TensorHandle, b: TensorHandle): TensorHandle;
  add(a: TensorHandle, b: TensorHandle): TensorHandle;
  sub(a: TensorHandle, b: TensorHandle): TensorHandle;
  mul(a: TensorHandle, b: TensorHandle): TensorHandle;
  div(a: TensorHandle, b: TensorHandle): TensorHandle;
  scale(a: TensorHandle, scalar: number): TensorHandle;

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

  conv1d(input: TensorHandle, kernel: TensorHandle, stride: number, padding: number): TensorHandle;
  conv2d(input: TensorHandle, kernel: TensorHandle, strides: [number, number], padding: 'same' | 'valid'): TensorHandle;
  depthwiseConv1d(input: TensorHandle, kernel: TensorHandle, stride: number, padding: number): TensorHandle;

  reshape(x: TensorHandle, shape: Shape): TensorHandle;
  transpose(x: TensorHandle, perm: number[]): TensorHandle;
  slice(x: TensorHandle, begin: number[], size: number[]): TensorHandle;
  concat(tensors: TensorHandle[], axis: number): TensorHandle;
  split(x: TensorHandle, numSplits: number, axis: number): TensorHandle[];
  gather(x: TensorHandle, indices: TensorHandle, axis: number): TensorHandle;

  getData(tensor: TensorHandle): Promise<Float32Array>;
  getShape(tensor: TensorHandle): Shape;
}
