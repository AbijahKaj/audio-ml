export type TensorHandle = unknown;
export type Shape = readonly number[];
export type Dtype = 'float32' | 'float16' | 'int32' | 'int64';
export type BackendKind = 'wasm' | 'webgpu' | 'webgl' | 'cpu';
