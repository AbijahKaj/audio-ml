/**
 * Integration test: load real parakeet-tdt_ctc-110m model and run inference.
 * Tests the full pipeline: SafeTensors → weight mapping → encoder → decoder → text
 */
import { readFileSync } from 'fs';
import '@tensorflow/tfjs';
import { TfjsBackend } from '../packages/asr/src/compute/TfjsBackend';
import { loadSafeTensors } from '../packages/asr/src/model/SafeTensorsLoader';
import { parseModelConfig } from '../packages/asr/src/model/ModelConfig';
import { mapWeights } from '../packages/asr/src/model/WeightMapper';
import { FeaturePipeline } from '../packages/asr/src/features/FeaturePipeline';
import { FastConformerEncoder } from '../packages/asr/src/encoder/FastConformerEncoder';
import { createDecoder } from '../packages/asr/src/decoder/createDecoder';
import { SentencePieceDecoder } from '../packages/asr/src/text/SentencePieceDecoder';

const MODEL_DIR = '/workspace/test_model';

async function main() {
  console.log('=== FastConformer ASR Integration Test ===\n');

  // Step 1: Initialize backend
  console.log('1. Initializing TF.js backend (CPU)...');
  const backend = new TfjsBackend();
  await backend.init('cpu');
  console.log('   ✓ Backend ready\n');

  // Step 2: Load config
  console.log('2. Loading model config...');
  const configJson = readFileSync(`${MODEL_DIR}/model_config.json`, 'utf-8');
  const config = parseModelConfig(configJson);
  console.log(`   Encoder layers: ${config.encoderLayers}`);
  console.log(`   d_model: ${config.dModel}`);
  console.log(`   Decoder type: ${config.decoderType}`);
  console.log(`   Vocab size: ${config.vocabSize}`);
  console.log(`   TDT durations: ${JSON.stringify(config.tdtDurations)}`);
  console.log('   ✓ Config loaded\n');

  // Step 3: Load weights
  console.log('3. Loading SafeTensors weights...');
  const weightBuffer = readFileSync(`${MODEL_DIR}/model.safetensors`).buffer;
  const startLoad = performance.now();
  const rawWeights = await loadSafeTensors(weightBuffer, backend);
  const loadTime = performance.now() - startLoad;
  console.log(`   Loaded ${rawWeights.size} tensors in ${loadTime.toFixed(0)}ms`);
  console.log('   ✓ Weights loaded\n');

  // Step 4: Map weights
  console.log('4. Mapping weights to model structure...');
  const startMap = performance.now();
  const modelWeights = mapWeights(rawWeights, config);
  const mapTime = performance.now() - startMap;
  console.log(`   Encoder: ${modelWeights.encoder.layers.length} layers`);
  console.log(`   Subsampling convs: ${modelWeights.encoder.subsampling.allConvWeights.length}`);
  console.log(`   Mapped in ${mapTime.toFixed(0)}ms`);
  console.log('   ✓ Weights mapped\n');

  // Step 5: Build pipeline
  console.log('5. Building feature pipeline...');
  const featurePipeline = new FeaturePipeline(config, backend);
  console.log(`   Window: ${config.windowSizeMs}ms, Hop: ${config.hopSizeMs}ms`);
  console.log(`   Mel bands: ${config.numMelBands}`);
  console.log('   ✓ Feature pipeline ready\n');

  // Step 6: Build encoder
  console.log('6. Building FastConformer encoder...');
  const encoder = new FastConformerEncoder(backend, modelWeights.encoder, config);
  console.log(`   ${encoder.numLayers} conformer blocks`);
  console.log('   ✓ Encoder ready\n');

  // Step 7: Build decoder
  console.log('7. Building TDT decoder...');
  const decoder = createDecoder(config, backend, modelWeights.decoder);
  console.log('   ✓ Decoder ready\n');

  // Step 8: Load vocab
  console.log('8. Loading vocabulary...');
  const vocabJson = readFileSync(`${MODEL_DIR}/vocab.json`, 'utf-8');
  const tokenizer = new SentencePieceDecoder(vocabJson);
  console.log(`   Vocab size: ${tokenizer.vocabSize}`);
  console.log('   ✓ Vocabulary loaded\n');

  // Step 9: Generate test audio (1 second of 440Hz sine wave at 16kHz)
  console.log('9. Generating test audio (440Hz sine, 1s at 16kHz)...');
  const sampleRate = 16000;
  const duration = 1.0;
  const numSamples = Math.round(sampleRate * duration);
  const audio = new Float32Array(numSamples);
  for (let i = 0; i < numSamples; i++) {
    audio[i] = 0.5 * Math.sin(2 * Math.PI * 440 * i / sampleRate);
  }
  console.log(`   ${numSamples} samples\n`);

  // Step 10: Extract features
  console.log('10. Extracting mel features...');
  const startFeat = performance.now();
  const melFeatures = featurePipeline.extractFeatures(audio);
  const featTime = performance.now() - startFeat;
  const melShape = backend.getShape(melFeatures);
  console.log(`   Mel shape: [${melShape.join(', ')}]`);
  console.log(`   Feature extraction: ${featTime.toFixed(0)}ms`);
  console.log('   ✓ Features extracted\n');

  // Step 11: Run encoder
  console.log('11. Running encoder...');
  const startEnc = performance.now();
  const encoderOutput = encoder.forward(melFeatures);
  const encTime = performance.now() - startEnc;
  const encShape = backend.getShape(encoderOutput);
  console.log(`   Encoder output shape: [${encShape.join(', ')}]`);
  console.log(`   Encoder time: ${encTime.toFixed(0)}ms`);
  console.log('   ✓ Encoder done\n');

  // Step 12: Run decoder
  console.log('12. Running TDT decoder...');
  const startDec = performance.now();
  const tokenIds = await decoder.decode(encoderOutput);
  const decTime = performance.now() - startDec;
  console.log(`   Token IDs: [${tokenIds.join(', ')}]`);
  console.log(`   Decoder time: ${decTime.toFixed(0)}ms`);
  console.log('   ✓ Decoder done\n');

  // Step 13: Decode text
  console.log('13. Decoding tokens to text...');
  const text = tokenizer.decode(tokenIds);
  console.log(`   Text: "${text}"`);
  console.log('   ✓ Text decoded\n');

  // Summary
  const totalTime = loadTime + mapTime + featTime + encTime + decTime;
  console.log('=== SUMMARY ===');
  console.log(`Model: parakeet-tdt_ctc-110m (${config.decoderType.toUpperCase()})`);
  console.log(`Audio: 1s @ 16kHz → ${melShape[1]} mel frames → ${encShape[1]} encoder frames`);
  console.log(`Output: "${text}" (${tokenIds.length} tokens)`);
  console.log(`Times: load=${loadTime.toFixed(0)}ms, feat=${featTime.toFixed(0)}ms, enc=${encTime.toFixed(0)}ms, dec=${decTime.toFixed(0)}ms`);
  console.log(`Total inference: ${(featTime + encTime + decTime).toFixed(0)}ms`);
  console.log('');
  console.log('✅ Integration test PASSED — full pipeline works end-to-end!');

  // Clean up
  backend.dispose(melFeatures);
  backend.dispose(encoderOutput);
}

main().catch(err => {
  console.error('❌ Test FAILED:', err.message);
  console.error(err.stack);
  process.exit(1);
});
