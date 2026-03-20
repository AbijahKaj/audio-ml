import type { ComputeBackend } from './Backend';
import type { TensorHandle } from './types';

export class ComputeScope {
  private tensors: TensorHandle[] = [];

  track<T extends TensorHandle>(t: T): T {
    this.tensors.push(t);
    return t;
  }

  keep(t: TensorHandle): void {
    const idx = this.tensors.indexOf(t);
    if (idx >= 0) {
      this.tensors.splice(idx, 1);
    }
  }

  dispose(backend: ComputeBackend): void {
    for (const t of this.tensors) {
      backend.dispose(t);
    }
    this.tensors = [];
  }
}
