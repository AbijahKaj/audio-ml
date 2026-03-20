import type { ComputeBackend } from '../compute/Backend';
import { ComputeScope } from '../compute/ComputeScope';
import type { TensorHandle } from '../compute/types';

/** PyTorch Linear weight layout [outFeatures, inFeatures] → y = x @ Wᵀ + b */
export function linearForward(
  backend: ComputeBackend,
  x: TensorHandle,
  weight: TensorHandle,
  bias: TensorHandle | null,
): TensorHandle {
  const scope = new ComputeScope();
  const wT = scope.track(backend.transpose(weight, [1, 0]));
  let y = scope.track(backend.matmul(x, wT));
  if (bias) {
    y = backend.add(y, bias);
  }
  scope.dispose(backend);
  return y;
}

export function embeddingRow(backend: ComputeBackend, table: TensorHandle, tokenId: number): TensorHandle {
  const idx = backend.tensor(new Int32Array([tokenId]), [1], 'int32');
  const row = backend.gather(table, idx, 0);
  backend.dispose(idx);
  return backend.reshape(row, [1, 1, -1]);
}
