import type { ComputeBackend } from './Backend.js';
import type { TensorHandle } from './types.js';

export class ComputeScope {
  private tensors: TensorHandle[] = [];

  track<T extends TensorHandle>(tensor: T): T {
    this.tensors.push(tensor);
    return tensor;
  }

  keep(tensor: TensorHandle): void {
    this.tensors = this.tensors.filter((entry) => entry !== tensor);
  }

  dispose(backend: ComputeBackend): void {
    backend.disposeMany(this.tensors);
    this.tensors = [];
  }
}
