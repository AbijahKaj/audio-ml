/**
 * Full benchmark: load real model, transcribe LibriSpeech audio,
 * compare against NeMo reference, and test streaming.
 */
import { readFileSync, readdirSync, existsSync } from 'fs';
import { createTfjsBackend } from './init-tfjs-backend.mjs';
import { TfjsBackend } from '../packages/asr/src/compute/TfjsBackend';
import { loadSafeTensors } from '../packages/asr/src/model/SafeTensorsLoader';
import { parseModelConfig } from '../packages/asr/src/model/ModelConfig';
import { mapWeights } from '../packages/asr/src/model/WeightMapper';
import { FeaturePipeline } from '../packages/asr/src/features/FeaturePipeline';
import { FastConformerEncoder } from '../packages/asr/src/encoder/FastConformerEncoder';
import { createDecoder } from '../packages/asr/src/decoder/createDecoder';
import { SentencePieceDecoder } from '../packages/asr/src/text/SentencePieceDecoder';
import { ChunkedInference } from '../packages/asr/src/streaming/ChunkedInference';

const MODEL_DIR = '/workspace/test_model';
const AUDIO_DIR = '/workspace/test_audio';

async function main() {
  console.log('╔══════════════════════════════════════════════════════════════╗');
  console.log('║  FastConformer ASR Benchmark — JS vs NeMo (Python/PyTorch)  ║');
  console.log('╚══════════════════════════════════════════════════════════════╝\n');

  // ── Load model ──
  console.log('Loading model...');
  const backend = await createTfjsBackend(TfjsBackend);

  const configJson = readFileSync(`${MODEL_DIR}/model_config.json`, 'utf-8');
  const config = parseModelConfig(configJson);

  const t0load = performance.now();
  const weightBuffer = readFileSync(`${MODEL_DIR}/model.safetensors`).buffer;
  const rawWeights = await loadSafeTensors(weightBuffer, backend);
  const modelWeights = mapWeights(rawWeights, config);
  const jsLoadTime = performance.now() - t0load;

  const featurePipeline = new FeaturePipeline(config, backend);
  const encoder = new FastConformerEncoder(backend, modelWeights.encoder, config);
  const decoder = createDecoder(config, backend, modelWeights.decoder);
  const vocabJson = readFileSync(`${MODEL_DIR}/vocab.json`, 'utf-8');
  const tokenizer = new SentencePieceDecoder(vocabJson);

  console.log(`  Model loaded in ${jsLoadTime.toFixed(0)}ms (JS/TF.js CPU)\n`);

  // ── Find audio files ──
  const audioFiles = readdirSync(AUDIO_DIR)
    .filter(f => /^libri_\d+\.f32$/.test(f))
    .sort();

  if (audioFiles.length === 0) {
    console.error('No audio files found! Run tools/prepare_test_audio.py first.');
    process.exit(1);
  }

  // ── Load NeMo reference results ──
  const nemoResults = {};
  const nemoBenchPath = `${AUDIO_DIR}/nemo_benchmark.json`;
  if (existsSync(nemoBenchPath)) {
    const nemoData = JSON.parse(readFileSync(nemoBenchPath, 'utf-8'));
    for (const r of nemoData) {
      nemoResults[r.file] = r;
    }
  }

  // ══════════════════════════════════════
  //  PART 1: OFFLINE TRANSCRIPTION
  // ══════════════════════════════════════
  console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
  console.log('  PART 1: Offline Transcription');
  console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n');

  const jsResults = [];

  for (const f32File of audioFiles) {
    const baseName = f32File.replace('.f32', '');
    const wavFile = `${baseName}.wav`;
    const metaPath = `${AUDIO_DIR}/${baseName}.json`;

    const meta = existsSync(metaPath) ? JSON.parse(readFileSync(metaPath, 'utf-8')) : {};
    const nemo = nemoResults[wavFile] || {};

    // Load raw float32 audio
    const audioBuffer = readFileSync(`${AUDIO_DIR}/${f32File}`);
    const audio = new Float32Array(audioBuffer.buffer, audioBuffer.byteOffset, audioBuffer.byteLength / 4);

    console.log(`─── ${wavFile} (${meta.duration_s?.toFixed(2) || '?'}s) ───`);
    console.log(`  Reference : "${meta.text || '?'}"`);
    if (nemo.nemo_output) {
      console.log(`  NeMo      : "${nemo.nemo_output}"`);
      console.log(`  NeMo time : ${nemo.nemo_avg_ms?.toFixed(0)}ms (RTF=${nemo.nemo_rtf?.toFixed(3)})`);
    }

    // JS inference
    const t0 = performance.now();
    const mel = featurePipeline.extractFeatures(audio);
    const featTime = performance.now() - t0;

    const t1 = performance.now();
    const encoded = encoder.forward(mel);
    const encTime = performance.now() - t1;

    const t2 = performance.now();
    const tokenIds = await decoder.decode(encoded);
    const decTime = performance.now() - t2;

    const text = tokenizer.decode(tokenIds);
    const totalTime = performance.now() - t0;
    const rtf = totalTime / ((meta.duration_s || 1) * 1000);

    console.log(`  JS output : "${text}"`);
    console.log(`  JS time   : ${totalTime.toFixed(0)}ms (feat=${featTime.toFixed(0)} enc=${encTime.toFixed(0)} dec=${decTime.toFixed(0)})`);
    console.log(`  JS RTF    : ${rtf.toFixed(3)}`);

    if (nemo.nemo_avg_ms) {
      const speedup = totalTime / nemo.nemo_avg_ms;
      console.log(`  Speed     : ${speedup.toFixed(1)}x slower than NeMo (CPU PyTorch)`);
    }

    const encShape = backend.getShape(encoded);
    console.log(`  Shapes    : mel=[1,${backend.getShape(mel).slice(1).join(',')}] → enc=[1,${encShape.slice(1).join(',')}]`);
    console.log(`  Tokens    : ${tokenIds.length} tokens`);
    console.log('');

    jsResults.push({
      file: wavFile,
      reference: meta.text || '',
      js_output: text,
      duration_s: meta.duration_s || 0,
      js_total_ms: totalTime,
      js_feat_ms: featTime,
      js_enc_ms: encTime,
      js_dec_ms: decTime,
      js_rtf: rtf,
      nemo_output: nemo.nemo_output || '',
      nemo_avg_ms: nemo.nemo_avg_ms || 0,
      nemo_rtf: nemo.nemo_rtf || 0,
      token_count: tokenIds.length,
    });

    backend.dispose(mel);
    backend.dispose(encoded);
  }

  // ══════════════════════════════════════
  //  PART 2: STREAMING TRANSCRIPTION
  // ══════════════════════════════════════
  console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
  console.log('  PART 2: Streaming Transcription');
  console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n');

  for (const f32File of audioFiles) {
    const baseName = f32File.replace('.f32', '');
    const wavFile = `${baseName}.wav`;
    const metaPath = `${AUDIO_DIR}/${baseName}.json`;
    const meta = existsSync(metaPath) ? JSON.parse(readFileSync(metaPath, 'utf-8')) : {};

    const audioBuffer = readFileSync(`${AUDIO_DIR}/${f32File}`);
    const audio = new Float32Array(audioBuffer.buffer, audioBuffer.byteOffset, audioBuffer.byteLength / 4);

    console.log(`─── ${wavFile} (${meta.duration_s?.toFixed(2) || '?'}s) — STREAMING ───`);

    // Streaming config: 500ms chunks
    const chunkSizeMs = 500;
    const chunkSizeSamples = Math.round(16000 * chunkSizeMs / 1000);

    const streaming = new ChunkedInference(
      backend, config, featurePipeline, encoder, decoder,
      tokenizer,
      { chunkSizeMs, maxContextFrames: 70, inputSampleRate: 16000 }
    );

    const t0stream = performance.now();
    let partialCount = 0;
    let lastPartialText = '';

    // Feed audio in chunks (simulating real-time mic input)
    for (let offset = 0; offset < audio.length; offset += chunkSizeSamples) {
      const end = Math.min(offset + chunkSizeSamples, audio.length);
      const chunk = audio.slice(offset, end);

      const result = await streaming.feedAudio(new Float32Array(chunk));
      if (result) {
        partialCount++;
        lastPartialText = result.text;
        const chunkTime = (offset / 16000).toFixed(1);
        if (partialCount <= 5 || partialCount % 3 === 0) {
          console.log(`  [${chunkTime}s] partial #${partialCount}: "${result.text}" (${result.latencyMs.toFixed(0)}ms)`);
        }
      }
    }

    // Flush remaining
    const finalResult = await streaming.flush();
    const streamTime = performance.now() - t0stream;

    console.log(`  [final] "${finalResult.text}"`);
    console.log(`  Stream time: ${streamTime.toFixed(0)}ms, ${partialCount} partials`);
    console.log(`  Stream RTF: ${(streamTime / ((meta.duration_s || 1) * 1000)).toFixed(3)}`);
    console.log('');

    streaming.reset();
  }

  // ══════════════════════════════════════
  //  SUMMARY
  // ══════════════════════════════════════
  console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
  console.log('  SUMMARY');
  console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n');

  console.log(`Model: nvidia/parakeet-tdt_ctc-110m (114M params, TDT)`);
  console.log(`JS Backend: TensorFlow.js CPU`);
  console.log(`NeMo Backend: PyTorch CPU\n`);

  console.log(`JS model load: ${jsLoadTime.toFixed(0)}ms\n`);

  console.log('┌──────────────┬──────────┬──────────┬──────────┬──────────┐');
  console.log('│ Audio        │ Duration │  NeMo ms │  JS ms   │ JS/NeMo  │');
  console.log('├──────────────┼──────────┼──────────┼──────────┼──────────┤');
  for (const r of jsResults) {
    const ratio = r.nemo_avg_ms > 0 ? (r.js_total_ms / r.nemo_avg_ms).toFixed(1) + 'x' : 'N/A';
    console.log(`│ ${r.file.padEnd(12)} │ ${r.duration_s.toFixed(1).padStart(6)}s │ ${r.nemo_avg_ms.toFixed(0).padStart(6)}ms │ ${r.js_total_ms.toFixed(0).padStart(6)}ms │ ${ratio.padStart(8)} │`);
  }
  console.log('└──────────────┴──────────┴──────────┴──────────┴──────────┘\n');

  console.log('Transcription comparison:');
  for (const r of jsResults) {
    console.log(`  ${r.file}:`);
    console.log(`    Reference: "${r.reference}"`);
    console.log(`    NeMo:      "${r.nemo_output}"`);
    console.log(`    JS:        "${r.js_output}"`);
    console.log('');
  }

  console.log('✅ Benchmark complete!');
}

main().catch(err => {
  console.error('❌ FAILED:', err.message);
  console.error(err.stack);
  process.exit(1);
});
