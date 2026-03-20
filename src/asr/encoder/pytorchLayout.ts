import type { TensorHandle } from '../compute/types';
import type { ComputeBackend } from '../compute/Backend';

/** PyTorch Conv2d [out, in, kH, kW] → tfjs conv2d [kH, kW, in, out] */
export function pytorchConv2dToTfjs(backend: ComputeBackend, w: TensorHandle): TensorHandle {
  return backend.transpose(w, [2, 3, 1, 0]);
}

/** PyTorch Conv1d [out, in, k] → tfjs conv1d NWC [k, in, out] */
export function pytorchConv1dToTfjs(backend: ComputeBackend, w: TensorHandle): TensorHandle {
  return backend.transpose(w, [2, 1, 0]);
}

/** PyTorch Linear [out, in] — already row-major; matmul uses x @ W^T */
export function pytorchLinearWeight(w: TensorHandle): TensorHandle {
  return w;
}

/** Depthwise Conv1d groups=d: PyTorch [d, 1, k] → tfjs depthwiseConv2d [1, k, d, 1] */
export function pytorchDepthwiseConv1dToTfjs(backend: ComputeBackend, w: TensorHandle): TensorHandle {
  const t = backend.transpose(w, [2, 1, 0]);
  const s = backend.getShape(t);
  return backend.reshape(t, [1, s[0]!, s[2]!, 1]);
}
