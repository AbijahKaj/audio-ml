import type { ComputeBackend, TensorHandle } from './index.js';

/**
 * Scope-based memory management for intermediate tensors.
 *
 * tfjs leaks memory if tensors aren't explicitly disposed.
 * ComputeScope tracks all tensors created in a compute block and disposes
 * them in bulk, except for tensors explicitly kept via `keep()`.
 *
 * Usage:
 *   const scope = new ComputeScope();
 *   const a = scope.track(backend.matmul(x, w));
 *   const b = scope.track(backend.add(a, bias));
 *   const out = backend.relu(b);   // not tracked — will be returned
 *   scope.dispose(backend);        // disposes a and b
 *   return out;
 */
export class ComputeScope {
  private readonly tensors: TensorHandle[] = [];

  track<T extends TensorHandle>(t: T): T {
    this.tensors.push(t);
    return t;
  }

  /** Remove a tensor from this scope (caller is responsible for disposing it). */
  keep(t: TensorHandle): void {
    const idx = this.tensors.indexOf(t);
    if (idx >= 0) this.tensors.splice(idx, 1);
  }

  dispose(backend: ComputeBackend): void {
    for (const t of this.tensors) backend.dispose(t);
    this.tensors.length = 0;
  }
}
