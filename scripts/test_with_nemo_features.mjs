/**
 * Test by feeding NeMo's exact mel features into our encoder+decoder,
 * bypassing the feature pipeline to isolate encoder/decoder correctness.
 */
import { readFileSync } from 'fs';
import { createTfjsBackend } from './init-tfjs-backend.mjs';
import { TfjsBackend } from '../packages/asr/src/compute/TfjsBackend';
import { loadSafeTensors } from '../packages/asr/src/model/SafeTensorsLoader';
import { parseModelConfig } from '../packages/asr/src/model/ModelConfig';
import { mapWeights } from '../packages/asr/src/model/WeightMapper';
import { FastConformerEncoder } from '../packages/asr/src/encoder/FastConformerEncoder';
import { createDecoder } from '../packages/asr/src/decoder/createDecoder';
import { SentencePieceDecoder } from '../packages/asr/src/text/SentencePieceDecoder';

const MODEL_DIR = '/workspace/test_model';
const AUDIO_DIR = '/workspace/test_audio';

async function main() {
  console.log('=== Test with NeMo Features (bypass JS feature pipeline) ===\n');

  const backend = await createTfjsBackend(TfjsBackend);

  const configJson = readFileSync(`${MODEL_DIR}/model_config.json`, 'utf-8');
  const config = parseModelConfig(configJson);

  const weightBuffer = readFileSync(`${MODEL_DIR}/model.safetensors`).buffer;
  const rawWeights = await loadSafeTensors(weightBuffer, backend);
  const modelWeights = mapWeights(rawWeights, config);

  const encoder = new FastConformerEncoder(backend, modelWeights.encoder, config);
  const decoder = createDecoder(config, backend, modelWeights.decoder);
  const vocabJson = readFileSync(`${MODEL_DIR}/vocab.json`, 'utf-8');
  const tokenizer = new SentencePieceDecoder(vocabJson);

  console.log('Model loaded.\n');

  // Load NeMo's exact features [1, 586, 80] (transposed from NeMo's [1, 80, 586])
  const featBuf = readFileSync(`${AUDIO_DIR}/libri_0_nemo_features.f32`);
  const featData = new Float32Array(featBuf.buffer, featBuf.byteOffset, featBuf.byteLength / 4);
  const T = 586;
  const F = 80;
  console.log(`Loaded NeMo features: ${featData.length} values → [1, ${T}, ${F}]`);
  console.log(`  First 10 values: [${Array.from(featData.slice(0, 10)).map(v => v.toFixed(4)).join(', ')}]`);
  
  const features = backend.tensor(featData, [1, T, F]);
  console.log(`  Feature tensor shape: [${backend.getShape(features).join(', ')}]`);
  console.log(`  min/max: check via stats...`);

  // Run encoder
  console.log('\nRunning encoder...');
  const t0 = performance.now();
  const encoded = encoder.forward(features);
  const encTime = performance.now() - t0;
  const encShape = backend.getShape(encoded);
  console.log(`  Encoder output: [${encShape.join(', ')}]`);
  console.log(`  Encoder time: ${encTime.toFixed(0)}ms`);

  // Check encoder output stats
  const encData = await backend.getData(encoded);
  let eMin = Infinity, eMax = -Infinity, eSum = 0;
  for (let i = 0; i < encData.length; i++) {
    if (encData[i] < eMin) eMin = encData[i];
    if (encData[i] > eMax) eMax = encData[i];
    eSum += encData[i];
  }
  console.log(`  Encoder stats: min=${eMin.toFixed(4)}, max=${eMax.toFixed(4)}, mean=${(eSum/encData.length).toFixed(4)}`);

  // Run decoder
  console.log('\nRunning decoder...');
  const t1 = performance.now();
  const tokenIds = await decoder.decode(encoded);
  const decTime = performance.now() - t1;
  console.log(`  Token IDs (${tokenIds.length}): [${tokenIds.slice(0, 30).join(', ')}...]`);
  console.log(`  Decoder time: ${decTime.toFixed(0)}ms`);

  const text = tokenizer.decode(tokenIds);
  console.log(`\n  JS output: "${text}"`);
  console.log(`  Expected:  "mister Quilter is the apostle of the middle classes, and we are glad to welcome his gospel."`);

  backend.dispose(features);
  backend.dispose(encoded);
}

main().catch(err => {
  console.error('FAILED:', err.message);
  console.error(err.stack);
  process.exit(1);
});
