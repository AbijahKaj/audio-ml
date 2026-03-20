import type { ComputeBackend } from '../compute/Backend';
import type { TensorHandle } from '../compute/types';

interface SafeTensorMeta {
  dtype: string;
  shape: number[];
  data_offsets: [number, number];
}

function bytesPerElement(dtype: string): number {
  switch (dtype) {
    case 'F32':
      return 4;
    case 'F16':
    case 'BF16':
      return 2;
    case 'I64':
      return 8;
    case 'I32':
      return 4;
    default:
      return 0;
  }
}

function decodeTensor(
  buffer: ArrayBuffer,
  dataStart: number,
  meta: SafeTensorMeta,
): { data: Float32Array | Int32Array; shape: number[]; dtype: 'float32' | 'int32' } {
  const [start, end] = meta.data_offsets;
  const byteLength = end - start;
  const offset = dataStart + start;
  const view = new DataView(buffer, offset, byteLength);

  if (meta.dtype === 'F32') {
    const n = byteLength / 4;
    const data = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      data[i] = view.getFloat32(i * 4, true);
    }
    return { data, shape: meta.shape, dtype: 'float32' };
  }

  if (meta.dtype === 'I32') {
    const n = byteLength / 4;
    const data = new Int32Array(n);
    for (let i = 0; i < n; i++) {
      data[i] = view.getInt32(i * 4, true);
    }
    return { data, shape: meta.shape, dtype: 'int32' };
  }

  if (meta.dtype === 'I64') {
    const n = byteLength / 8;
    const data = new Int32Array(n);
    for (let i = 0; i < n; i++) {
      const lo = view.getUint32(i * 8, true);
      const hi = view.getInt32(i * 8 + 4, true);
      if (hi !== 0 && hi !== -1) {
        throw new Error('SafeTensors I64 value out of int32 range');
      }
      data[i] = lo;
    }
    return { data, shape: meta.shape, dtype: 'int32' };
  }

  if (meta.dtype === 'F16' || meta.dtype === 'BF16') {
    const n = byteLength / 2;
    const data = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const u16 = view.getUint16(i * 2, true);
      data[i] = float16ToFloat32(u16);
    }
    return { data, shape: meta.shape, dtype: 'float32' };
  }

  throw new Error(`Unsupported SafeTensors dtype: ${meta.dtype}`);
}

/** Half-precision to float32 (IEEE 754 binary16). */
function float16ToFloat32(h: number): number {
  const s = (h & 0x8000) >> 15;
  const e = (h & 0x7c00) >> 10;
  const f = h & 0x03ff;
  if (e === 0) {
    return (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
  }
  if (e === 31) {
    return f !== 0 ? NaN : s ? -Infinity : Infinity;
  }
  const exp = e - 15;
  const mant = 1 + f / 1024;
  return (s ? -1 : 1) * Math.pow(2, exp) * mant;
}

export async function loadSafeTensors(
  source: string | ArrayBuffer,
  backend: ComputeBackend,
): Promise<Map<string, TensorHandle>> {
  const buffer =
    typeof source === 'string'
      ? await (await fetch(source)).arrayBuffer()
      : source;

  const headerLen = Number(new DataView(buffer).getBigUint64(0, true));
  const headerBytes = new Uint8Array(buffer, 8, headerLen);
  let headerJson = new TextDecoder().decode(headerBytes).replace(/\0/g, '').trim();
  const brace = headerJson.lastIndexOf('}');
  if (brace >= 0) {
    headerJson = headerJson.slice(0, brace + 1);
  }
  const header = JSON.parse(headerJson) as Record<string, SafeTensorMeta | Record<string, unknown>>;

  const dataStart = 8 + headerLen;
  const tensors = new Map<string, TensorHandle>();

  for (const [name, meta] of Object.entries(header)) {
    if (name === '__metadata__' || !meta || typeof meta !== 'object' || !('dtype' in meta)) {
      continue;
    }
    const m = meta as SafeTensorMeta;
    const bpe = bytesPerElement(m.dtype);
    const [s, e] = m.data_offsets;
    if (bpe === 0) {
      console.warn(`Skipping tensor ${name}: unsupported dtype ${m.dtype}`);
      continue;
    }
    if ((e - s) % bpe !== 0) {
      throw new Error(`Invalid byte length for ${name}`);
    }
    const { data, shape, dtype } = decodeTensor(buffer, dataStart, m);
    tensors.set(name, backend.tensor(data, shape, dtype));
  }

  return tensors;
}
