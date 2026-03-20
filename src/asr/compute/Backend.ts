import type { TensorHandle, Shape } from './types.js';

/**
 * Backend abstraction for tensor operations.
 * All model code goes through this interface so the compute engine is swappable.
 */
export interface ComputeBackend {
  // ── Tensor creation ──────────────────────────────────────────────────────────
  tensor(data: Float32Array | Int32Array, shape: Shape): TensorHandle;
  zeros(shape: Shape): TensorHandle;
  ones(shape: Shape): TensorHandle;
  scalar(value: number): TensorHandle;
  dispose(t: TensorHandle): void;

  // ── Core arithmetic ──────────────────────────────────────────────────────────
  matmul(a: TensorHandle, b: TensorHandle, transposeA?: boolean, transposeB?: boolean): TensorHandle;
  add(a: TensorHandle, b: TensorHandle): TensorHandle;
  sub(a: TensorHandle, b: TensorHandle): TensorHandle;
  mul(a: TensorHandle, b: TensorHandle): TensorHandle;
  div(a: TensorHandle, b: TensorHandle): TensorHandle;
  scale(a: TensorHandle, s: number): TensorHandle;
  sqrt(x: TensorHandle): TensorHandle;
  exp(x: TensorHandle): TensorHandle;
  log(x: TensorHandle): TensorHandle;
  pow(x: TensorHandle, exp: number): TensorHandle;

  // ── Reductions ────────────────────────────────────────────────────────────────
  sum(x: TensorHandle, axis: number, keepDims?: boolean): TensorHandle;
  mean(x: TensorHandle, axis: number, keepDims?: boolean): TensorHandle;
  max(x: TensorHandle, axis: number, keepDims?: boolean): TensorHandle;
  softmax(x: TensorHandle, axis?: number): TensorHandle;
  argmax(x: TensorHandle, axis: number): TensorHandle;

  // ── Normalization ─────────────────────────────────────────────────────────────
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

  // ── Activations ───────────────────────────────────────────────────────────────
  relu(x: TensorHandle): TensorHandle;
  silu(x: TensorHandle): TensorHandle;
  gelu(x: TensorHandle): TensorHandle;
  sigmoid(x: TensorHandle): TensorHandle;
  tanh(x: TensorHandle): TensorHandle;

  // ── Convolutions ──────────────────────────────────────────────────────────────
  conv1d(
    input: TensorHandle,
    kernel: TensorHandle,
    bias: TensorHandle | null,
    stride: number,
    padding: 'same' | 'valid',
  ): TensorHandle;
  conv2d(
    input: TensorHandle,
    kernel: TensorHandle,
    bias: TensorHandle | null,
    strides: [number, number],
    padding: 'same' | 'valid',
  ): TensorHandle;
  depthwiseConv1d(
    input: TensorHandle,
    kernel: TensorHandle,
    stride: number,
    padding: 'same' | 'valid',
  ): TensorHandle;

  // ── Shape manipulation ────────────────────────────────────────────────────────
  reshape(x: TensorHandle, shape: Shape): TensorHandle;
  transpose(x: TensorHandle, perm: number[]): TensorHandle;
  expandDims(x: TensorHandle, axis: number): TensorHandle;
  squeeze(x: TensorHandle, axis?: number[]): TensorHandle;
  slice(x: TensorHandle, begin: number[], size: number[]): TensorHandle;
  concat(tensors: TensorHandle[], axis: number): TensorHandle;
  split(x: TensorHandle, numOrSizeSplits: number | number[], axis: number): TensorHandle[];
  gather(x: TensorHandle, indices: TensorHandle, axis?: number): TensorHandle;
  pad(x: TensorHandle, paddings: Array<[number, number]>): TensorHandle;
  tile(x: TensorHandle, reps: number[]): TensorHandle;
  stack(tensors: TensorHandle[], axis?: number): TensorHandle;
  unstack(x: TensorHandle, axis?: number): TensorHandle[];

  // ── Masking / fill ────────────────────────────────────────────────────────────
  fill(shape: Shape, value: number): TensorHandle;
  where(condition: TensorHandle, x: TensorHandle, y: TensorHandle): TensorHandle;

  // ── Data transfer ─────────────────────────────────────────────────────────────
  getData(t: TensorHandle): Promise<Float32Array>;
  getInt32Data(t: TensorHandle): Promise<Int32Array>;
  getShape(t: TensorHandle): Shape;
}
