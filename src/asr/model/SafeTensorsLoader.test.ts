import { describe, expect, it } from 'vitest';
import { TfjsBackend } from '../compute/TfjsBackend';
import { loadSafeTensors } from './SafeTensorsLoader';

function buildMinimalSafeTensorsF32(name: string, data: Float32Array, shape: number[]): ArrayBuffer {
  const header = {
    [name]: {
      dtype: 'F32',
      shape,
      data_offsets: [0, data.byteLength],
    },
  };
  const headerJson = JSON.stringify(header);
  const enc = new TextEncoder().encode(headerJson);
  const pad = (8 - (enc.length % 8)) % 8;
  const headerPadded = new Uint8Array(enc.length + pad);
  headerPadded.set(enc);
  for (let i = enc.length; i < headerPadded.length; i++) {
    headerPadded[i] = 0x20;
  }
  const headerLen = BigInt(headerPadded.length);
  const total = 8 + headerPadded.length + data.byteLength;
  const buf = new ArrayBuffer(total);
  const view = new DataView(buf);
  view.setBigUint64(0, headerLen, true);
  new Uint8Array(buf, 8, headerPadded.length).set(headerPadded);
  new Float32Array(buf, 8 + headerPadded.length, data.length).set(data);
  return buf;
}

describe('loadSafeTensors', () => {
  it('loads a single F32 tensor', async () => {
    const backend = new TfjsBackend('cpu');
    const data = new Float32Array([1, 2, 3, 4]);
    const buf = buildMinimalSafeTensorsF32('test.w', data, [2, 2]);
    const map = await loadSafeTensors(buf, backend);
    expect(map.has('test.w')).toBe(true);
    const t = map.get('test.w')!;
    const out = (await backend.getData(t)) as Float32Array;
    expect([...out]).toEqual([1, 2, 3, 4]);
    backend.dispose(t);
  });
});
