import type { ComputeBackend } from '../compute/Backend';
import type { TensorHandle } from '../compute/types';
import type { LinearWeights } from '../model/types';
import { pytorchLinearWeight } from './pytorchLayout';

export function linearForward(
  backend: ComputeBackend,
  x: TensorHandle,
  w: LinearWeights,
): TensorHandle {
  const wt = backend.transpose(pytorchLinearWeight(w.weight), [1, 0]);
  const out = backend.matmul(x, wt);
  backend.dispose(wt);
  return backend.add(out, w.bias);
}