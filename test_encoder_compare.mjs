/**
 * Compare our encoder output to NeMo's using NeMo's exact features.
 */
import { readFileSync } from 'fs';
import '@tensorflow/tfjs-backend-cpu';
import { TfjsBackend } from './src/asr/compute/TfjsBackend';
import { loadSafeTensors } from './src/asr/model/SafeTensorsLoader';
import { parseModelConfig } from './src/asr/model/ModelConfig';
import { mapWeights } from './src/asr/model/WeightMapper';
import { FastConformerEncoder } from './src/asr/encoder/FastConformerEncoder';

const MODEL_DIR = '/workspace/test_model';
const AUDIO_DIR = '/workspace/test_audio';

async function main() {
  console.log('=== Encoder Comparison ===\n');

  const backend = new TfjsBackend();
  await backend.init('cpu');

  const configJson = readFileSync(`${MODEL_DIR}/model_config.json`, 'utf-8');
  const config = parseModelConfig(configJson);
  const weightBuffer = readFileSync(`${MODEL_DIR}/model.safetensors`).buffer;
  const rawWeights = await loadSafeTensors(weightBuffer, backend);
  const modelWeights = mapWeights(rawWeights, config);
  const encoder = new FastConformerEncoder(backend, modelWeights.encoder, config);

  // Load NeMo features [1, 586, 80]
  const featBuf = readFileSync(`${AUDIO_DIR}/libri_0_nemo_features.f32`);
  const featData = new Float32Array(featBuf.buffer, featBuf.byteOffset, featBuf.byteLength / 4);
  const features = backend.tensor(featData, [1, 586, 80]);

  // Load NeMo encoder output [1, 74, 512]
  const nemoEncBuf = readFileSync(`${AUDIO_DIR}/libri_0_nemo_encoded.f32`);
  const nemoEncData = new Float32Array(nemoEncBuf.buffer, nemoEncBuf.byteOffset, nemoEncBuf.byteLength / 4);

  // Run our encoder
  console.log('Running JS encoder...');
  const encoded = encoder.forward(features);
  const jsEncData = await backend.getData(encoded);
  const jsShape = backend.getShape(encoded);
  console.log(`  JS encoder output: [${jsShape.join(', ')}]`);
  console.log(`  NeMo encoder output: [1, 74, 512] (${nemoEncData.length} values)`);

  // Compare
  const T = Math.min(jsShape[1], 74);
  const D = 512;
  let sumAbsDiff = 0;
  let maxDiff = 0;
  let count = 0;
  
  for (let t = 0; t < T; t++) {
    for (let d = 0; d < D; d++) {
      const jsVal = jsEncData[t * D + d];
      const nemoVal = nemoEncData[t * D + d];
      const diff = Math.abs(jsVal - nemoVal);
      sumAbsDiff += diff;
      if (diff > maxDiff) maxDiff = diff;
      count++;
    }
  }

  const meanAbsDiff = sumAbsDiff / count;
  console.log(`\n  Comparison (first ${T} frames):`);
  console.log(`  Mean absolute diff: ${meanAbsDiff.toFixed(6)}`);
  console.log(`  Max absolute diff: ${maxDiff.toFixed(6)}`);

  // Show first frame comparison
  console.log(`\n  First frame comparison (first 10 dims):`);
  console.log(`    JS:   [${Array.from(jsEncData.slice(0, 10)).map(v => v.toFixed(4)).join(', ')}]`);
  console.log(`    NeMo: [${Array.from(nemoEncData.slice(0, 10)).map(v => v.toFixed(4)).join(', ')}]`);

  // Cosine similarity of first frame
  let dotProd = 0, normA = 0, normB = 0;
  for (let d = 0; d < D; d++) {
    dotProd += jsEncData[d] * nemoEncData[d];
    normA += jsEncData[d] * jsEncData[d];
    normB += nemoEncData[d] * nemoEncData[d];
  }
  const cosSim = dotProd / (Math.sqrt(normA) * Math.sqrt(normB));
  console.log(`\n  Cosine similarity (first frame): ${cosSim.toFixed(6)}`);

  backend.dispose(features);
  backend.dispose(encoded);
}

main().catch(err => {
  console.error('FAILED:', err.message);
  console.error(err.stack);
  process.exit(1);
});
