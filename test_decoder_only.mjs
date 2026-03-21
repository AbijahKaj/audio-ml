/**
 * Test decoder with NeMo's exact encoder output to isolate decoder correctness.
 */
import { readFileSync } from 'fs';
import '@tensorflow/tfjs-backend-cpu';
import { TfjsBackend } from './src/asr/compute/TfjsBackend';
import { loadSafeTensors } from './src/asr/model/SafeTensorsLoader';
import { parseModelConfig } from './src/asr/model/ModelConfig';
import { mapWeights } from './src/asr/model/WeightMapper';
import { createDecoder } from './src/asr/decoder/createDecoder';
import { SentencePieceDecoder } from './src/asr/text/SentencePieceDecoder';

const MODEL_DIR = '/workspace/test_model';
const AUDIO_DIR = '/workspace/test_audio';

async function main() {
  console.log('=== Decoder-Only Test (NeMo encoder output) ===\n');

  const backend = new TfjsBackend();
  await backend.init('cpu');

  const configJson = readFileSync(`${MODEL_DIR}/model_config.json`, 'utf-8');
  const config = parseModelConfig(configJson);
  
  const weightBuffer = readFileSync(`${MODEL_DIR}/model.safetensors`).buffer;
  const rawWeights = await loadSafeTensors(weightBuffer, backend);
  const modelWeights = mapWeights(rawWeights, config);
  
  const decoder = createDecoder(config, backend, modelWeights.decoder);
  const vocabJson = readFileSync(`${MODEL_DIR}/vocab.json`, 'utf-8');
  const tokenizer = new SentencePieceDecoder(vocabJson);

  // Load NeMo's exact encoder output [1, 74, 512]
  const encBuf = readFileSync(`${AUDIO_DIR}/libri_0_nemo_encoded.f32`);
  const encData = new Float32Array(encBuf.buffer, encBuf.byteOffset, encBuf.byteLength / 4);
  const T = 74;
  const D = 512;
  console.log(`Loaded NeMo encoder output: ${encData.length} values → [1, ${T}, ${D}]`);
  console.log(`  First 5: [${Array.from(encData.slice(0, 5)).map(v => v.toFixed(4)).join(', ')}]`);
  
  const encoderOutput = backend.tensor(encData, [1, T, D]);

  // Run decoder
  console.log('\nRunning TDT decoder...');
  console.log(`  blankId = ${config.vocabSize - 1}`);
  console.log(`  vocabSize = ${config.vocabSize}`);
  console.log(`  durations = ${JSON.stringify(config.tdtDurations)}`);
  
  const t0 = performance.now();
  const tokenIds = await decoder.decode(encoderOutput);
  const decTime = performance.now() - t0;
  
  console.log(`  Token IDs (${tokenIds.length}): [${tokenIds.slice(0, 50).join(', ')}]`);
  console.log(`  Decoder time: ${decTime.toFixed(0)}ms`);
  
  const text = tokenizer.decode(tokenIds);
  console.log(`\n  JS decoder output: "${text}"`);
  console.log(`  Expected (NeMo):   "mister Quilter is the apostle of the middle classes, and we are glad to welcome his gospel."`);
  
  // Debug: check first frame's joint output
  console.log('\n--- Debug: first frame joint network ---');
  const encFrame = backend.slice(encoderOutput, [0, 0, 0], [1, 1, D]);
  
  // Get prediction for blank token
  const predNet = modelWeights.decoder.prediction;
  const embedding = predNet.embedding;
  const embShape = backend.getShape(embedding);
  console.log(`  Embedding shape: [${embShape.join(', ')}]`);
  
  // Check joint weight shapes
  const jw = modelWeights.decoder.joint;
  console.log(`  Joint encoder proj: [${backend.getShape(jw.encoderProj.weight).join(', ')}]`);
  console.log(`  Joint pred proj: [${backend.getShape(jw.predictionProj.weight).join(', ')}]`);
  console.log(`  Joint output proj: [${backend.getShape(jw.outputProj.weight).join(', ')}]`);

  backend.dispose(encoderOutput);
  backend.dispose(encFrame);
}

main().catch(err => {
  console.error('FAILED:', err.message);
  console.error(err.stack);
  process.exit(1);
});
