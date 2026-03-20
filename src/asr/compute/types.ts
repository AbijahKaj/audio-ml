/**
 * Core types for the compute backend abstraction.
 * TensorHandle is opaque — it wraps a tf.Tensor internally but callers never see that.
 */

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type TensorHandle = any;
export type Shape = readonly number[];
export type Dtype = 'float32' | 'float16' | 'int32' | 'int64';
