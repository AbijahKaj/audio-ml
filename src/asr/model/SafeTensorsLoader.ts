import type { ComputeBackend } from '../compute/Backend';
import type { TensorHandle } from '../compute/types';

interface SafeTensorMeta {
  dtype: string;
  shape: number[];
  data_offsets: [number, number];
}

function dtypeToByteSize(dtype: string): number {
  switch (dtype) {
    case 'F32': return 4;
    case 'F16': return 2;
    case 'I32': return 4;
    case 'I64': return 8;
    case 'BF16': return 2;
    default: return 4;
  }
}

function convertToFloat32(data: ArrayBuffer, dtype: string, shape: number[]): Float32Array {
  const numElements = shape.reduce((a, b) => a * b, 1);

  switch (dtype) {
    case 'F32':
      return new Float32Array(data);
    case 'F16': {
      const uint16 = new Uint16Array(data);
      const result = new Float32Array(numElements);
      for (let i = 0; i < numElements; i++) {
        result[i] = float16ToFloat32(uint16[i]);
      }
      return result;
    }
    case 'BF16': {
      const bf16 = new Uint16Array(data);
      const result = new Float32Array(numElements);
      for (let i = 0; i < numElements; i++) {
        result[i] = bfloat16ToFloat32(bf16[i]);
      }
      return result;
    }
    case 'I32':
      return new Float32Array(new Int32Array(data));
    case 'I64': {
      const view = new DataView(data);
      const result = new Float32Array(numElements);
      for (let i = 0; i < numElements; i++) {
        result[i] = Number(view.getBigInt64(i * 8, true));
      }
      return result;
    }
    default:
      return new Float32Array(data);
  }
}

function float16ToFloat32(h: number): number {
  const sign = (h >> 15) & 0x1;
  const exponent = (h >> 10) & 0x1f;
  const fraction = h & 0x3ff;

  if (exponent === 0) {
    if (fraction === 0) return sign ? -0 : 0;
    const f = fraction / 1024;
    return (sign ? -1 : 1) * f * Math.pow(2, -14);
  }
  if (exponent === 0x1f) {
    return fraction ? NaN : (sign ? -Infinity : Infinity);
  }
  return (sign ? -1 : 1) * Math.pow(2, exponent - 15) * (1 + fraction / 1024);
}

function bfloat16ToFloat32(bf: number): number {
  const buf = new ArrayBuffer(4);
  const view = new DataView(buf);
  view.setUint16(0, bf, false);
  view.setUint16(2, 0, false);
  return view.getFloat32(0, false);
}

export async function loadSafeTensors(
  source: string | ArrayBuffer,
  backend: ComputeBackend
): Promise<Map<string, TensorHandle>> {
  const buffer = typeof source === 'string'
    ? await (await fetch(source)).arrayBuffer()
    : source;

  const headerLenView = new DataView(buffer, 0, 8);
  const headerLen = Number(headerLenView.getBigUint64(0, true));
  const headerJson = new TextDecoder().decode(new Uint8Array(buffer, 8, headerLen));
  const header: Record<string, SafeTensorMeta | Record<string, string>> = JSON.parse(headerJson);

  const dataStart = 8 + headerLen;
  const tensors = new Map<string, TensorHandle>();

  for (const [name, meta] of Object.entries(header)) {
    if (name === '__metadata__') continue;

    const tensorMeta = meta as SafeTensorMeta;
    const [start, end] = tensorMeta.data_offsets;
    const rawData = buffer.slice(dataStart + start, dataStart + end);
    const _byteSize = dtypeToByteSize(tensorMeta.dtype);
    const float32Data = convertToFloat32(rawData, tensorMeta.dtype, tensorMeta.shape);
    tensors.set(name, backend.tensor(float32Data, tensorMeta.shape));
  }

  return tensors;
}

export async function loadSafeTensorsFromBuffer(
  buffer: ArrayBuffer,
  backend: ComputeBackend
): Promise<Map<string, TensorHandle>> {
  return loadSafeTensors(buffer, backend);
}
