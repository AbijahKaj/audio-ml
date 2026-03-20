import type { ComputeBackend } from '../compute/Backend';
import type { Dtype, TensorHandle } from '../compute/types';

export interface SafeTensorHeaderEntry {
  dtype: string;
  shape: number[];
  data_offsets: [number, number];
}

function toSupportedDtype(dtype: string): Dtype {
  switch (dtype) {
    case 'I32':
      return 'int32';
    case 'I64':
      return 'int64';
    case 'F16':
      return 'float16';
    case 'F32':
    default:
      return 'float32';
  }
}

function sliceData(buffer: ArrayBuffer, offset: number, length: number, dtype: Dtype): Float32Array | Int32Array {
  const slice = buffer.slice(offset, offset + length);
  if (dtype === 'int32' || dtype === 'int64') {
    return new Int32Array(slice);
  }

  if (dtype === 'float16') {
    const view = new Uint16Array(slice);
    const output = new Float32Array(view.length);
    for (let index = 0; index < view.length; index += 1) {
      const value = view[index];
      const sign = (value & 0x8000) >> 15;
      const exponent = (value & 0x7c00) >> 10;
      const fraction = value & 0x03ff;

      if (exponent === 0) {
        output[index] = (sign ? -1 : 1) * Math.pow(2, -14) * (fraction / 1024);
      } else if (exponent === 0x1f) {
        output[index] = fraction ? Number.NaN : sign ? Number.NEGATIVE_INFINITY : Number.POSITIVE_INFINITY;
      } else {
        output[index] = (sign ? -1 : 1) * Math.pow(2, exponent - 15) * (1 + fraction / 1024);
      }
    }
    return output;
  }

  return new Float32Array(slice);
}

async function resolveSource(source: string | ArrayBuffer): Promise<ArrayBuffer> {
  if (typeof source !== 'string') {
    return source;
  }

  const response = await fetch(source);
  if (!response.ok) {
    throw new Error(`Failed to fetch SafeTensors file: ${response.status} ${response.statusText}`);
  }
  return response.arrayBuffer();
}

export async function loadSafeTensors(
  source: string | ArrayBuffer,
  backend: ComputeBackend,
): Promise<Map<string, TensorHandle>> {
  const buffer = await resolveSource(source);
  const view = new DataView(buffer);
  const headerLength = Number(view.getBigUint64(0, true));
  const headerJson = new TextDecoder().decode(new Uint8Array(buffer, 8, headerLength));
  const header = JSON.parse(headerJson) as Record<string, SafeTensorHeaderEntry>;
  const dataStart = 8 + headerLength;
  const tensors = new Map<string, TensorHandle>();

  for (const [name, entry] of Object.entries(header)) {
    if (name === '__metadata__') {
      continue;
    }

    const dtype = toSupportedDtype(entry.dtype);
    const [start, end] = entry.data_offsets;
    const data = sliceData(buffer, dataStart + start, end - start, dtype);
    tensors.set(name, backend.tensor(data, entry.shape, dtype));
  }

  return tensors;
}
