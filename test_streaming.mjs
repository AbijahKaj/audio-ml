/**
 * Streaming inference test — simulates the browser demo's audio pipeline.
 *
 * Feeds real LibriSpeech audio in small chunks (matching the demo's 320ms
 * chunk size) through ChunkedInference, exactly as processFrameAsync does.
 */
import { readFileSync } from 'fs';
import '@tensorflow/tfjs-backend-cpu';
import { TfjsBackend } from './src/asr/compute/TfjsBackend';
import { loadSafeTensors } from './src/asr/model/SafeTensorsLoader';
import { parseModelConfig } from './src/asr/model/ModelConfig';
import { mapWeights } from './src/asr/model/WeightMapper';
import { FeaturePipeline } from './src/asr/features/FeaturePipeline';
import { FastConformerEncoder } from './src/asr/encoder/FastConformerEncoder';
import { createDecoder } from './src/asr/decoder/createDecoder';
import { SentencePieceDecoder } from './src/asr/text/SentencePieceDecoder';
import { ChunkedInference } from './src/asr/streaming/ChunkedInference';

const MODEL_DIR = '/workspace/test_model';
const AUDIO_DIR = '/workspace/test_audio';

async function main() {
  console.log('=== Streaming Inference Test ===\n');

  const backend = new TfjsBackend();
  await backend.init('cpu');

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

  console.log(`Model: ${config.decoderType.toUpperCase()}, ${config.encoderLayers} layers, d=${config.dModel}`);
  console.log(`Conv kernel: ${config.convKernelSize}, subsampling: ${config.subsamplingFactor}x\n`);

  const chunked = new ChunkedInference(
    backend, config, featurePipeline, encoder, decoder, tokenizer,
    { chunkSizeMs: 320, maxContextFrames: 70, inputSampleRate: 16000 },
  );

  // --- Test 1: feed small synthetic chunks (like mic input) ---
  console.log('--- Test 1: Small synthetic chunks (320ms each) ---');
  const sr = 16000;
  const chunkSamples = Math.round(sr * 320 / 1000);
  console.log(`Chunk size: ${chunkSamples} samples (320ms)\n`);

  for (let i = 0; i < 8; i++) {
    const pcm = new Float32Array(chunkSamples);
    for (let j = 0; j < chunkSamples; j++) {
      pcm[j] = 0.3 * Math.sin(2 * Math.PI * 440 * (i * chunkSamples + j) / sr);
    }
    const t0 = performance.now();
    const result = await chunked.feedAudio(pcm);
    const elapsed = performance.now() - t0;
    if (result) {
      console.log(`  Chunk ${i}: "${result.text}" (${elapsed.toFixed(0)}ms)`);
    } else {
      console.log(`  Chunk ${i}: buffering... (${elapsed.toFixed(0)}ms)`);
    }
  }

  const synthFinal = await chunked.flush();
  console.log(`  Final: "${synthFinal.text}" (${synthFinal.decoderType})\n`);
  chunked.reset();

  // --- Test 2: feed real LibriSpeech audio ---
  console.log('--- Test 2: Real LibriSpeech audio (streaming) ---');
  const audioMeta = JSON.parse(readFileSync(`${AUDIO_DIR}/libri_0.json`, 'utf-8'));
  const audioBuf = readFileSync(`${AUDIO_DIR}/libri_0.f32`);
  const audio = new Float32Array(audioBuf.buffer, audioBuf.byteOffset, audioBuf.byteLength / 4);
  console.log(`Audio: ${audio.length} samples (${(audio.length / sr).toFixed(1)}s)`);
  console.log(`Expected: "${audioMeta.text}"\n`);

  let offset = 0;
  let chunkIdx = 0;
  const startTime = performance.now();
  while (offset < audio.length) {
    const end = Math.min(offset + chunkSamples, audio.length);
    const pcm = audio.subarray(offset, end);
    offset = end;

    const result = await chunked.feedAudio(pcm);
    if (result && result.text.trim()) {
      console.log(`  Chunk ${chunkIdx}: "${result.text}" (${result.latencyMs.toFixed(0)}ms)`);
    }
    chunkIdx++;
  }

  const finalResult = await chunked.flush();
  const totalMs = performance.now() - startTime;
  console.log(`\n  Final text: "${finalResult.text}"`);
  console.log(`  Expected:   "${audioMeta.text}"`);
  console.log(`  Total time: ${totalMs.toFixed(0)}ms for ${(audio.length / sr).toFixed(1)}s audio`);
  console.log(`  RTF: ${(totalMs / 1000 / (audio.length / sr)).toFixed(2)}x`);

  console.log('\n✅ Streaming test PASSED — no crashes!');
}

main().catch(err => {
  console.error('❌ Test FAILED:', err.message);
  console.error(err.stack);
  process.exit(1);
});
