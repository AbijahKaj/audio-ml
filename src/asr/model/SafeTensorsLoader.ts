import type { ComputeBackend } from '../compute/Backend';
import type { TensorHandle } from '../compute/types';

const HEADER_LEN_BYTES = 8;

function parseHeaderLength(buffer: ArrayBuffer): number {
  const view = new DataView(buffer);
  return Number(view.getBigUint64(0, true));
}

interface TensorMeta {
  dtype: string;
  shape: number[];
  data_offsets: [number, number];
}

function readTensorData(
  buffer: ArrayBuffer,
  dataStart: number,
  meta: TensorMeta,
): Float32Array {
  const [start, end] = meta.data_offsets;
  const byteLength = end - start;
  const slice = buffer.slice(dataStart + start, dataStart + start + byteLength);

  if (meta.dtype === 'F32') {
    return new Float32Array(slice);
  }

  if (meta.dtype === 'F16') {
    const u16 = new Uint16Array(slice);
    const out = new Float32Array(u16.length);
    for (let i = 0; i < u16.length; i++) {
      out[i] = float16ToFloat32(u16[i]!);
    }
    return out;
  }

  throw new Error(`Unsupported SafeTensors dtype for ASR: ${meta.dtype}`);
}

/** IEEE 754 half-precision to float32 */
function float16ToFloat32(h: number): number {
  const s = (h & 0x8000) >> 15;
  const e = (h & 0x7c00) >> 10;
  const f = h & 0x03ff;
  if (e === 0) {
    return (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
  }
  if (e === 31) {
    return f ? NaN : (s ? -Infinity : Infinity);
  }
  return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
}

/**
 * Load a SafeTensors file into backend tensors (float32).
 */
export async function loadSafeTensors(
  source: string | ArrayBuffer,
  backend: ComputeBackend,
): Promise<Map<string, TensorHandle>> {
  const buffer =
    typeof source === 'string'
      ? await (await fetch(source)).arrayBuffer()
      : source;

  const headerLen = parseHeaderLength(buffer);
  const headerJson = new TextDecoder().decode(new Uint8Array(buffer, HEADER_LEN_BYTES, headerLen));
  const header = JSON.parse(headerJson) as Record<string, TensorMeta | Record<string, unknown>>;

  const dataStart = HEADER_LEN_BYTES + headerLen;
  const tensors = new Map<string, TensorHandle>();

  for (const [name, meta] of Object.entries(header)) {
    if (name === '__metadata__' || typeof meta !== 'object' || !('data_offsets' in meta)) {
      continue;
    }
    const m = meta as TensorMeta;
    const data = readTensorData(buffer, dataStart, m);
    tensors.set(name, backend.tensor(data, m.shape));
  }

  return tensors;
}

/**
 * Load from a Node.js Buffer or Uint8Array (no fetch).
 */
export function loadSafeTensorsSync(
  buffer: ArrayBuffer,
  backend: ComputeBackend,
): Map<string, TensorHandle> {
  const headerLen = parseHeaderLength(buffer);
  const headerJson = new TextDecoder().decode(new Uint8Array(buffer, HEADER_LEN_BYTES, headerLen));
  const header = JSON.parse(headerJson) as Record<string, TensorMeta | Record<string, unknown>>;

  const dataStart = HEADER_LEN_BYTES + headerLen;
  const tensors = new Map<string, TensorHandle>();

  for (const [name, meta] of Object.entries(header)) {
    if (name === '__metadata__' || typeof meta !== 'object' || !('data_offsets' in meta)) {
      continue;
    }
    const m = meta as TensorMeta;
    const data = readTensorData(buffer, dataStart, m);
    tensors.set(name, backend.tensor(data, m.shape));
  }

  return tensors;
}