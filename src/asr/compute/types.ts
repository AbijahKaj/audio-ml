/**
 * Opaque tensor handle used by compute backends.
 */
export type TensorHandle = unknown;

/**
 * Tensor shape in row-major order.
 */
export type Shape = readonly number[];

/**
 * Supported tensor dtypes.
 */
export type Dtype = 'float32' | 'float16' | 'int32' | 'int64' | 'bool';
