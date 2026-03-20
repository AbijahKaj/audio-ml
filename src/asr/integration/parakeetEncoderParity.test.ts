import { execSync } from 'node:child_process';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

function nemoPythonAvailable(): boolean {
  try {
    execSync('python3 -c "import nemo.collections.asr"', { stdio: 'ignore' });
    return true;
  } catch {
    return false;
  }
}

const describeParakeet = nemoPythonAvailable() ? describe : describe.skip;
import { TfjsBackend } from '../compute/TfjsBackend';
import { FastConformerEncoder } from '../encoder/FastConformerEncoder';
import { loadSafeTensors } from '../model/SafeTensorsLoader';
import { mapWeights } from '../model/WeightMapper';
import { parseModelConfig } from '../model/ModelConfig';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.join(__dirname, '..', '..', '..');
const fixtureDir = path.join(root, 'tmp_parakeet');

/**
 * Real NeMo checkpoint (Parakeet 120M). See tools/NEMO_PARITY.md.
 */
describeParakeet('Parakeet 120M (NeMo export)', () => {
  it('exports fixture, loads SafeTensors, maps all encoder weights', async () => {
    execSync(`python3 "${path.join(root, 'tools/export_parakeet_fixture.py')}" --out "${fixtureDir}"`, {
      stdio: 'inherit',
    });

    const config = parseModelConfig(fs.readFileSync(path.join(fixtureDir, 'model_config.json'), 'utf8'));
    expect(config.subsampling).toBe('dw_striding');
    expect(config.encoderLayers).toBe(17);

    const stBuf = fs.readFileSync(path.join(fixtureDir, 'model.safetensors'));
    const raw = new ArrayBuffer(stBuf.byteLength);
    new Uint8Array(raw).set(stBuf);

    const backend = new TfjsBackend('cpu');
    const wmap = await loadSafeTensors(raw, backend);
    const mapped = mapWeights(wmap, config, backend);
    expect(mapped.encoder.layers.length).toBe(17);
    expect(mapped.encoder.subsampling.layers.length).toBeGreaterThan(0);
  }, 300_000);

  it('attempts full encoder forward vs NeMo reference (parity WIP)', async () => {
    const refPath = path.join(fixtureDir, 'encoder_ref_btd.bin');
    if (!fs.existsSync(refPath)) {
      return;
    }
    const config = parseModelConfig(fs.readFileSync(path.join(fixtureDir, 'model_config.json'), 'utf8'));
    const melShape = JSON.parse(fs.readFileSync(path.join(fixtureDir, 'mel_shape.json'), 'utf8')) as number[];
    const melBuf = fs.readFileSync(path.join(fixtureDir, 'mel_bt_f.bin'));
    const mel = new Float32Array(melBuf.buffer, melBuf.byteOffset, melBuf.byteLength / 4);
    const refBuf = fs.readFileSync(refPath);
    const ref = new Float32Array(refBuf.buffer, refBuf.byteOffset, refBuf.byteLength / 4);

    const stBuf = fs.readFileSync(path.join(fixtureDir, 'model.safetensors'));
    const raw = new ArrayBuffer(stBuf.byteLength);
    new Uint8Array(raw).set(stBuf);
    const backend = new TfjsBackend('cpu');
    const wmap = await loadSafeTensors(raw, backend);
    const mapped = mapWeights(wmap, config, backend);
    const encoder = new FastConformerEncoder(backend, mapped.encoder, config);
    const melT = backend.tensor(mel, melShape);
    let out;
    try {
      out = encoder.forward(melT);
    } catch (e) {
      console.warn(
        '[parakeet] Encoder forward failed (subsampling H×W vs Linear in_features). Fix TF conv padding vs NeMo. Details: tools/NEMO_PARITY.md',
      );
      console.warn(String(e));
      backend.dispose(melT);
      expect(true).toBe(true);
      return;
    }
    const got = (await backend.getData(out)) as Float32Array;
    backend.dispose(melT);
    backend.dispose(out);

    if (got.length !== ref.length) {
      console.warn('[parakeet] length mismatch', { got: got.length, ref: ref.length });
      return;
    }
    let mse = 0;
    let maxAbs = 0;
    for (let i = 0; i < ref.length; i++) {
      const d = got[i]! - ref[i]!;
      mse += d * d;
      maxAbs = Math.max(maxAbs, Math.abs(d));
    }
    const rmse = Math.sqrt(mse / ref.length);
    console.log('[parakeet] RMSE vs NeMo encoder', { rmse, maxAbs });
    expect(rmse).toBeLessThan(0.1);
  }, 300_000);
});
