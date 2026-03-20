import { describe, expect, it } from 'vitest';
import { TfjsBackend } from '../../src/asr/compute/TfjsBackend';
import { loadSafeTensorsSync } from '../../src/asr/model/SafeTensorsLoader';

describe('SafeTensorsLoader', () => {
  it('parses a minimal F32 tensor', async () => {
    await TfjsBackend.init('cpu');
    const backend = new TfjsBackend();
    const data = new Float32Array([1, 2, 3, 4]);
    const header = {
      w: {
        dtype: 'F32',
        shape: [2, 2],
        data_offsets: [0, 16],
      },
    };
    const headerJson = JSON.stringify(header);
    const headerBytes = new TextEncoder().encode(headerJson);
    const headerLen = BigInt(headerBytes.length);
    const dataStart = 8 + headerBytes.length;
    const buf = new ArrayBuffer(dataStart + 16);
    const view = new DataView(buf);
    view.setBigUint64(0, headerLen, true);
    new Uint8Array(buf, 8, headerBytes.length).set(headerBytes);
    for (let i = 0; i < 4; i++) {
      view.setFloat32(dataStart + i * 4, data[i]!, true);
    }

    const map = loadSafeTensorsSync(buf, backend);
    expect(map.has('w')).toBe(true);
    const t = map.get('w')!;
    const out = await backend.getData(t);
    expect(Array.from(out)).toEqual([1, 2, 3, 4]);
    backend.dispose(t);
  });
});
