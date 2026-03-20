import type { ComputeBackend } from './Backend';
import type { TensorHandle } from './types';

/**
 * Tracks temporary tensors and disposes them as a group.
 */
export class ComputeScope {
  private tensors: TensorHandle[] = [];

  track<T extends TensorHandle>(tensor: T): T {
    this.tensors.push(tensor);
    return tensor;
  }

  keep(tensor: TensorHandle): TensorHandle {
    const index = this.tensors.indexOf(tensor);
    if (index >= 0) {
      this.tensors.splice(index, 1);
    }
    return tensor;
  }

  dispose(backend: ComputeBackend): void {
    for (const tensor of this.tensors) {
      backend.dispose(tensor);
    }
    this.tensors = [];
  }
}
