import type { ComputeBackend } from '../compute/Backend';
import type { TensorHandle } from '../compute/types';

const INF_VAL = 10000.0;

/**
 * NeMo-style relative positional encoding buffer (sin/cos), shape [1, 2*length-1, dModel].
 */
export function extendRelativePe(
  backend: ComputeBackend,
  length: number,
  dModel: number,
): TensorHandle {
  const posLen = 2 * length - 1;
  const positions = new Float32Array(posLen);
  for (let i = 0; i < posLen; i++) {
    positions[i] = length - 1 - i;
  }

  const pe = new Float32Array(posLen * dModel);
  const half = Math.ceil(dModel / 2);
  const divTerm = new Float32Array(half);
  for (let i = 0; i < half; i++) {
    divTerm[i] = Math.exp((i * 2) * -(Math.log(INF_VAL) / dModel));
  }

  for (let pos = 0; pos < posLen; pos++) {
    const p = positions[pos]!;
    const row = pos * dModel;
    for (let i = 0; i < half; i++) {
      const angle = p * divTerm[i]!;
      pe[row + 2 * i] = Math.sin(angle);
      if (2 * i + 1 < dModel) {
        pe[row + 2 * i + 1] = Math.cos(angle);
      }
    }
  }

  return backend.tensor(pe, [1, posLen, dModel]);
}

/**
 * Slice positional embedding for a sequence of length `seqLen` (NeMo RelPositionalEncoding.forward).
 */
export function sliceRelPosEmb(
  backend: ComputeBackend,
  peFull: TensorHandle,
  seqLen: number,
  cacheLen = 0,
): TensorHandle {
  const inputLen = seqLen + cacheLen;
  const peShape = backend.getShape(peFull);
  const peLen = peShape[1]!;
  const centerPos = Math.floor(peLen / 2) + 1;
  const startPos = centerPos - inputLen;
  const endPos = centerPos + inputLen - 1;
  const size = endPos - startPos;
  return backend.slice(peFull, [0, startPos, 0], [1, size, peShape[2]!]);
}