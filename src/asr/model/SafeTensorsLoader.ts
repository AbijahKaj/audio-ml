import type { ComputeBackend, TensorHandle } from '../compute/index.js';

/**
 * Load a SafeTensors file and return a map from tensor name → TensorHandle.
 *
 * SafeTensors format (https://huggingface.co/docs/safetensors):
 *   bytes 0–7    : uint64 little-endian  — header byte length N
 *   bytes 8..8+N : UTF-8 JSON string     — header metadata
 *   bytes 8+N..  : raw tensor data (concatenated, no padding)
 *
 * Header JSON structure:
 *   { "__metadata__": {...}, "weight_name": { "dtype": "F32", "shape": [...], "data_offsets": [start, end] }, ... }
 */
export async function loadSafeTensors(
  source: string | ArrayBuffer,
  backend: ComputeBackend,
): Promise<Map<string, TensorHandle>> {
  const buffer =
    typeof source === 'string'
      ? await (await fetch(source)).arrayBuffer()
      : source;

  const view = new DataView(buffer);

  // Read 8-byte header length (little-endian uint64)
  // JavaScript numbers are 53-bit integers, so BigInt is used here.
  const headerLen = Number(view.getBigUint64(0, true));

  const headerJson = new TextDecoder().decode(
    new Uint8Array(buffer, 8, headerLen),
  );

  type TensorMeta = {
    dtype: string;
    shape: number[];
    data_offsets: [number, number];
  };
  const header = JSON.parse(headerJson) as Record<string, TensorMeta | Record<string, string>>;

  const dataStart = 8 + headerLen;
  const tensors = new Map<string, TensorHandle>();

  for (const [name, meta] of Object.entries(header)) {
    if (name === '__metadata__') continue;

    const { dtype, shape, data_offsets } = meta as TensorMeta;
    const [start, end] = data_offsets;
    const byteSlice = buffer.slice(dataStart + start, dataStart + end);

    let tensorData: Float32Array | Int32Array;

    switch (dtype) {
      case 'F32':
        tensorData = new Float32Array(byteSlice);
        break;
      case 'F16': {
        // Convert float16 → float32
        tensorData = float16ToFloat32(new Uint16Array(byteSlice));
        break;
      }
      case 'I32':
        tensorData = new Int32Array(byteSlice);
        break;
      case 'I64': {
        // Convert int64 → int32 (values should be small enough for our use)
        const big = new BigInt64Array(byteSlice);
        tensorData = new Int32Array(big.length);
        for (let i = 0; i < big.length; i++) {
          tensorData[i] = Number(big[i]);
        }
        break;
      }
      case 'BF16': {
        tensorData = bfloat16ToFloat32(new Uint16Array(byteSlice));
        break;
      }
      default:
        console.warn(`SafeTensors: unsupported dtype ${dtype} for ${name}, skipping`);
        continue;
    }

    tensors.set(name, backend.tensor(tensorData, shape));
  }

  return tensors;
}

function float16ToFloat32(u16: Uint16Array): Float32Array {
  const out = new Float32Array(u16.length);
  for (let i = 0; i < u16.length; i++) {
    const h = u16[i];
    const sign = (h & 0x8000) >> 15;
    const exp = (h & 0x7c00) >> 10;
    const frac = h & 0x03ff;

    if (exp === 0) {
      out[i] = (sign ? -1 : 1) * Math.pow(2, -14) * (frac / 1024);
    } else if (exp === 31) {
      out[i] = frac ? NaN : (sign ? -Infinity : Infinity);
    } else {
      out[i] = (sign ? -1 : 1) * Math.pow(2, exp - 15) * (1 + frac / 1024);
    }
  }
  return out;
}

function bfloat16ToFloat32(u16: Uint16Array): Float32Array {
  const out = new Float32Array(u16.length);
  const buf = new ArrayBuffer(4);
  const view = new DataView(buf);
  for (let i = 0; i < u16.length; i++) {
    view.setUint32(0, u16[i] << 16, false);
    out[i] = view.getFloat32(0, false);
  }
  return out;
}
