import type { ComputeBackend } from '../compute/Backend';
import type { Dtype, TensorHandle } from '../compute/types';

interface SafeTensorMeta {
  dtype: string;
  shape: number[];
  data_offsets: [number, number];
}

type SafeTensorHeader = Record<string, SafeTensorMeta | Record<string, string>>;

interface FetchResponseLike {
  arrayBuffer(): Promise<ArrayBuffer>;
}

type FetchLike = (input: string) => Promise<FetchResponseLike>;

function getGlobalFetch(): FetchLike | undefined {
  return (globalThis as unknown as { fetch?: FetchLike }).fetch;
}

function toArrayBuffer(source: Uint8Array): ArrayBuffer {
  const copy = new Uint8Array(source.byteLength);
  copy.set(source);
  return copy.buffer;
}

function decodeF16(value: number): number {
  const sign = (value & 0x8000) ? -1 : 1;
  const exponent = (value >> 10) & 0x1f;
  const fraction = value & 0x03ff;
  if (exponent === 0) {
    return sign * Math.pow(2, -14) * (fraction / 1024);
  }
  if (exponent === 31) {
    return fraction === 0 ? sign * Infinity : Number.NaN;
  }
  return sign * Math.pow(2, exponent - 15) * (1 + fraction / 1024);
}

function normalizeDtype(dtype: string): Dtype {
  if (dtype === 'F16' || dtype === 'F32') return 'float32';
  if (dtype === 'I64' || dtype === 'I32') return 'int32';
  if (dtype === 'BOOL') return 'bool';
  return 'float32';
}

function decodeData(
  rawBuffer: ArrayBuffer,
  dataStart: number,
  meta: SafeTensorMeta,
): Float32Array | Int32Array | Uint8Array {
  const [start, end] = meta.data_offsets;
  const begin = dataStart + start;
  const length = end - start;
  const view = new DataView(rawBuffer, begin, length);

  if (meta.dtype === 'F32') {
    return new Float32Array(rawBuffer, begin, length / Float32Array.BYTES_PER_ELEMENT).slice();
  }

  if (meta.dtype === 'F16') {
    const out = new Float32Array(length / 2);
    for (let i = 0; i < out.length; i++) {
      out[i] = decodeF16(view.getUint16(i * 2, true));
    }
    return out;
  }

  if (meta.dtype === 'I64') {
    const out = new Int32Array(length / 8);
    for (let i = 0; i < out.length; i++) {
      out[i] = Number(view.getBigInt64(i * 8, true));
    }
    return out;
  }

  if (meta.dtype === 'I32') {
    return new Int32Array(rawBuffer, begin, length / Int32Array.BYTES_PER_ELEMENT).slice();
  }

  if (meta.dtype === 'BOOL') {
    return new Uint8Array(rawBuffer, begin, length).slice();
  }

  throw new Error(`Unsupported SafeTensors dtype: ${meta.dtype}`);
}

async function resolveSource(source: string | ArrayBuffer | Uint8Array): Promise<ArrayBuffer> {
  if (source instanceof ArrayBuffer) {
    return source;
  }

  if (source instanceof Uint8Array) {
    return toArrayBuffer(source);
  }

  const fetchFn = getGlobalFetch();
  if (!fetchFn) {
    throw new Error(
      'SafeTensors string source requires global fetch(). Provide an ArrayBuffer in this environment.',
    );
  }

  const response = await fetchFn(source);
  return response.arrayBuffer();
}

export async function loadSafeTensors(
  source: string | ArrayBuffer | Uint8Array,
  backend: ComputeBackend,
): Promise<Map<string, TensorHandle>> {
  const buffer = await resolveSource(source);
  const headerLength = Number(new DataView(buffer).getBigUint64(0, true));
  const headerText = new TextDecoder().decode(new Uint8Array(buffer, 8, headerLength));
  const header = JSON.parse(headerText) as SafeTensorHeader;
  const dataStart = 8 + headerLength;
  const tensors = new Map<string, TensorHandle>();

  for (const [name, rawMeta] of Object.entries(header)) {
    if (name === '__metadata__') continue;
    const meta = rawMeta as SafeTensorMeta;
    const data = decodeData(buffer, dataStart, meta);
    tensors.set(name, backend.tensor(data, meta.shape, normalizeDtype(meta.dtype)));
  }

  return tensors;
}
