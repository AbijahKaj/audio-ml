/**
 * Test with speech-like audio (mix of frequencies simulating speech formants)
 * and also with silence to verify different model behaviors.
 */
import { readFileSync } from 'fs';
import { createTfjsBackend } from './init-tfjs-backend.mjs';
import { TfjsBackend } from '../packages/asr/src/compute/TfjsBackend';
import { loadSafeTensors } from '../packages/asr/src/model/SafeTensorsLoader';
import { parseModelConfig } from '../packages/asr/src/model/ModelConfig';
import { mapWeights } from '../packages/asr/src/model/WeightMapper';
import { FeaturePipeline } from '../packages/asr/src/features/FeaturePipeline';
import { FastConformerEncoder } from '../packages/asr/src/encoder/FastConformerEncoder';
import { createDecoder } from '../packages/asr/src/decoder/createDecoder';
import { SentencePieceDecoder } from '../packages/asr/src/text/SentencePieceDecoder';

const MODEL_DIR = '/workspace/test_model';

async function transcribe(encoder, decoder, tokenizer, featurePipeline, backend, audio, label) {
  console.log(`\n--- ${label} ---`);
  const mel = featurePipeline.extractFeatures(audio);
  const melShape = backend.getShape(mel);
  console.log(`  Mel: [${melShape.join(', ')}]`);

  const start = performance.now();
  const encoded = encoder.forward(mel);
  const encShape = backend.getShape(encoded);
  console.log(`  Encoder: [${encShape.join(', ')}]`);

  const tokenIds = await decoder.decode(encoded);
  const text = tokenizer.decode(tokenIds);
  const elapsed = performance.now() - start;

  console.log(`  Tokens (${tokenIds.length}): [${tokenIds.slice(0, 20).join(', ')}${tokenIds.length > 20 ? '...' : ''}]`);
  console.log(`  Text: "${text}"`);
  console.log(`  Time: ${elapsed.toFixed(0)}ms`);

  backend.dispose(mel);
  backend.dispose(encoded);
  return text;
}

async function main() {
  console.log('=== Speech-Like Audio Test ===');

  const backend = await createTfjsBackend(TfjsBackend);

  const configJson = readFileSync(`${MODEL_DIR}/model_config.json`, 'utf-8');
  const config = parseModelConfig(configJson);
  const weightBuffer = readFileSync(`${MODEL_DIR}/model.safetensors`).buffer;
  const rawWeights = await loadSafeTensors(weightBuffer, backend);
  const modelWeights = mapWeights(rawWeights, config);

  const featurePipeline = new FeaturePipeline(config, backend);
  const encoder = new FastConformerEncoder(backend, modelWeights.encoder, config);
  const decoder = createDecoder(config, backend, modelWeights.decoder);
  const vocabJson = readFileSync(`${MODEL_DIR}/vocab.json`, 'utf-8');
  const tokenizer = new SentencePieceDecoder(vocabJson);

  console.log('Model loaded successfully.');

  const sr = 16000;

  // Test 1: Silence
  const silence = new Float32Array(sr * 2); // 2 seconds of silence
  await transcribe(encoder, decoder, tokenizer, featurePipeline, backend, silence, 'Silence (2s)');

  // Test 2: White noise
  const noise = new Float32Array(sr * 2);
  for (let i = 0; i < noise.length; i++) noise[i] = (Math.random() - 0.5) * 0.1;
  await transcribe(encoder, decoder, tokenizer, featurePipeline, backend, noise, 'White noise (2s)');

  // Test 3: Speech-like formants (simulate vowel sounds)
  const vowelLike = new Float32Array(sr * 2);
  for (let i = 0; i < vowelLike.length; i++) {
    const t = i / sr;
    // F0 (fundamental) + F1, F2 formants typical of speech
    const f0 = 150; // typical male pitch
    const f1 = 700; // first formant
    const f2 = 1200; // second formant
    vowelLike[i] = 0.3 * Math.sin(2 * Math.PI * f0 * t)
                 + 0.2 * Math.sin(2 * Math.PI * f1 * t)
                 + 0.1 * Math.sin(2 * Math.PI * f2 * t);
    // Add some amplitude modulation (like syllables)
    vowelLike[i] *= 0.5 + 0.5 * Math.sin(2 * Math.PI * 4 * t);
  }
  await transcribe(encoder, decoder, tokenizer, featurePipeline, backend, vowelLike, 'Speech-like formants (2s)');

  // Test 4: Short audio (0.5s)
  const shortAudio = new Float32Array(sr * 0.5);
  for (let i = 0; i < shortAudio.length; i++) {
    shortAudio[i] = 0.3 * Math.sin(2 * Math.PI * 200 * i / sr);
  }
  await transcribe(encoder, decoder, tokenizer, featurePipeline, backend, shortAudio, 'Short audio (0.5s)');

  console.log('\n✅ All tests completed successfully!');
}

main().catch(err => {
  console.error('❌ Test FAILED:', err.message);
  console.error(err.stack);
  process.exit(1);
});
