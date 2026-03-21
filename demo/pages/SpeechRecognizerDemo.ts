/**
 * Speech Recognizer (ASR) Demo Page
 *
 * Loads a FastConformer model, streams mic / file audio through the
 * FastConformerASR and displays live transcription results.
 */

import { FastConformerASR, type ASRResult, Endpointer } from '@audio-ml/asr';
import { AudioInput } from '../components/AudioInput';
import { AudioInputUI } from '../components/AudioInputUI';

const INPUT_SAMPLE_RATE = 16_000;

interface ModelSpec {
  label: string;
  configUrl: string;
  weightsUrl: string;
  vocabUrl: string;
  description: string;
  sizeMB: number;
}

/**
 * HuggingFace-hosted models.
 * Converted from NeMo checkpoints via tools/export_nemo_to_safetensors.py
 *
 * To use your own model, run the export script and push to HF:
 *   python tools/export_nemo_to_safetensors.py \
 *       --model nvidia/parakeet-tdt_ctc-110m \
 *       --output-dir exported/parakeet-tdt-110m
 *   huggingface-cli upload YOUR_USER/parakeet-tdt-110m-web exported/parakeet-tdt-110m .
 *
 * Then add the HF resolve URLs below.
 */
function hfModel(repo: string): { config: string; weights: string; vocab: string } {
  const base = `https://huggingface.co/${repo}/resolve/main`;
  return {
    config: `${base}/model_config.json`,
    weights: `${base}/model.safetensors`,
    vocab: `${base}/vocab.json`,
  };
}

const HF_PARAKEET_TDT_110M = hfModel('AbijahKaj/parakeet-tdt-110m-web');
const HF_PARAKEET_RNNT_120M = hfModel('AbijahKaj/parakeet-rnnt-120m-web');
const HF_FASTCONFORMER_TDT_LARGE = hfModel('AbijahKaj/fastconformer-tdt-large-web');

const MODELS: Record<string, ModelSpec> = {
  parakeetTdt110m: {
    label: 'Parakeet TDT 110M',
    configUrl: HF_PARAKEET_TDT_110M.config,
    weightsUrl: HF_PARAKEET_TDT_110M.weights,
    vocabUrl: HF_PARAKEET_TDT_110M.vocab,
    description: 'English, 110M params, TDT decoder — fast, browser-optimized',
    sizeMB: 220,
  },
  parakeetRnnt120m: {
    label: 'Parakeet RNNT 120M (Streaming)',
    configUrl: HF_PARAKEET_RNNT_120M.config,
    weightsUrl: HF_PARAKEET_RNNT_120M.weights,
    vocabUrl: HF_PARAKEET_RNNT_120M.vocab,
    description: 'English, 120M params, RNNT decoder — streaming-optimized with end-of-utterance',
    sizeMB: 220,
  },
  fastconformerTdtLarge: {
    label: 'FastConformer TDT Large',
    configUrl: HF_FASTCONFORMER_TDT_LARGE.config,
    weightsUrl: HF_FASTCONFORMER_TDT_LARGE.weights,
    vocabUrl: HF_FASTCONFORMER_TDT_LARGE.vocab,
    description: 'English, 115M params, TDT decoder — high accuracy offline model',
    sizeMB: 218,
  },
};

/* ------------------------------------------------------------------ */
/*  Cache API helper — avoids re-downloading ~120 MB on every reload  */
/* ------------------------------------------------------------------ */

const MODEL_CACHE_NAME = 'asr-models-v1';

async function fetchCached(url: string, onProgress?: (pct: number) => void): Promise<ArrayBuffer> {
  const cache = await caches.open(MODEL_CACHE_NAME);
  const cached = await cache.match(url);
  if (cached) return cached.arrayBuffer();

  const response = await fetch(url);
  if (!response.ok) throw new Error(`Failed to fetch ${url}: ${response.status}`);
  const ct = response.headers.get('content-type') ?? '';
  if (ct.includes('text/html')) {
    throw new Error(
      `Model file not found at ${url} — check HuggingFace repo URL or run tools/export_nemo_to_safetensors.py`,
    );
  }

  const contentLength = Number(response.headers.get('content-length') ?? 0);
  const reader = response.body?.getReader();
  if (!reader || contentLength === 0) {
    const buf = await response.clone().arrayBuffer();
    await cache.put(url, response);
    return buf;
  }

  const chunks: Uint8Array[] = [];
  let received = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    onProgress?.(contentLength > 0 ? received / contentLength : 0);
  }

  const buf = new Uint8Array(received);
  let pos = 0;
  for (const c of chunks) {
    buf.set(c, pos);
    pos += c.length;
  }

  const cacheResponse = new Response(buf, {
    headers: { 'Content-Type': 'application/octet-stream' },
  });
  await cache.put(url, cacheResponse);
  return buf.buffer;
}

async function fetchTextCached(url: string): Promise<string> {
  const cache = await caches.open(MODEL_CACHE_NAME);
  const cached = await cache.match(url);
  if (cached) return cached.text();

  const response = await fetch(url);
  if (!response.ok) throw new Error(`Failed to fetch ${url}: ${response.status}`);
  const text = await response.text();
  if (text.trimStart().startsWith('<')) {
    throw new Error(
      `Model file not found at ${url} — check HuggingFace repo URL or run tools/export_nemo_to_safetensors.py`,
    );
  }
  await cache.put(url, new Response(text, { headers: { 'Content-Type': 'application/json' } }));
  return text;
}

/* ------------------------------------------------------------------ */
/*  Demo page                                                          */
/* ------------------------------------------------------------------ */

export function createSpeechRecognizerDemo(container: HTMLElement): () => void {
  let recognizer: FastConformerASR | null = null;
  let audioInput: AudioInput | null = null;
  let destroyed = false;

  // ---- layout ----
  const wrapper = el('div', 'asr-wrapper');

  const header = el('div', 'asr-header');
  header.innerHTML = `<h2 class="asr-title">Speech Recognition</h2>
    <p class="asr-subtitle">FastConformer RNNT / TDT &mdash; runs entirely in the browser via TensorFlow.js</p>
    <p class="asr-subtitle" style="font-size:0.85rem;opacity:0.7;margin-top:0.25rem">
      Record or upload audio &rarr; streaming preview during recording &rarr; accurate offline transcription on stop
    </p>`;
  wrapper.appendChild(header);

  // model loading section
  const loadSection = el('div', 'asr-load-section');

  const modelSelect = document.createElement('select');
  modelSelect.className = 'asr-select';
  for (const [key, spec] of Object.entries(MODELS)) {
    const opt = document.createElement('option');
    opt.value = key;
    opt.textContent = `${spec.label} (~${spec.sizeMB} MB)`;
    opt.title = spec.description;
    modelSelect.appendChild(opt);
  }

  const backendSelect = document.createElement('select');
  backendSelect.className = 'asr-select';
  // GPU only: pure JS / WASM CPU backends make this model unusable in the browser (long freezes).
  for (const [value, label] of [['webgpu', 'WebGPU'], ['webgl', 'WebGL']] as const) {
    const opt = document.createElement('option');
    opt.value = value;
    opt.textContent = label;
    if (value === 'webgpu' && !('gpu' in navigator)) opt.disabled = true;
    backendSelect.appendChild(opt);
  }
  backendSelect.value = 'gpu' in navigator ? 'webgpu' : 'webgl';

  const backendHint = el('div', 'asr-backend-hint');
  backendHint.style.cssText =
    'font-size:0.8rem;opacity:0.75;margin:0.35rem 0 0;line-height:1.35;max-width:36rem';
  backendHint.textContent =
    'CPU and WASM backends are not offered here — they block the page for too long on this model. Use WebGPU where available, otherwise WebGL.';

  const loadBtn = document.createElement('button');
  loadBtn.className = 'asr-load-btn';
  loadBtn.textContent = 'Load Model';

  const progressBar = el('div', 'asr-progress-bar');
  const progressFill = el('div', 'asr-progress-fill');
  progressBar.appendChild(progressFill);

  const statusLabel = el('div', 'asr-status');
  statusLabel.textContent = 'Model not loaded';

  loadSection.append(modelSelect, backendSelect, backendHint, loadBtn, progressBar, statusLabel);
  wrapper.appendChild(loadSection);

  // audio input section (hidden until model loaded)
  const audioSection = el('div', 'asr-audio-section');
  audioSection.style.display = 'none';
  wrapper.appendChild(audioSection);

  // transcript area
  const transcriptSection = el('div', 'asr-transcript-section');
  transcriptSection.style.display = 'none';

  const transcriptTitle = el('h3', 'asr-transcript-title');
  transcriptTitle.textContent = 'Transcription';
  transcriptSection.appendChild(transcriptTitle);

  const partialLine = el('div', 'asr-partial');
  partialLine.textContent = '';
  transcriptSection.appendChild(partialLine);

  const transcriptLog = el('div', 'asr-transcript-log');
  transcriptSection.appendChild(transcriptLog);

  const latencyLabel = el('div', 'asr-latency');
  transcriptSection.appendChild(latencyLabel);

  wrapper.appendChild(transcriptSection);

  // explainer
  const explainer = el('div', 'app-explainer');
  explainer.innerHTML = `
    <h2 class="app-explainer-title">How It Works</h2>
    <p class="app-explainer-description">
      This demo runs NVIDIA's FastConformer ASR model <strong>entirely in the browser</strong>
      using <code>TensorFlow.js</code>. No audio leaves your device.
    </p>
    <div class="app-explainer-phases">
      <div class="app-explainer-phase">
        <h4>1. Feature Extraction</h4>
        <p>Raw PCM is converted to 80-band log-mel spectrograms (25 ms window, 10 ms hop).</p>
      </div>
      <div class="app-explainer-phase">
        <h4>2. FastConformer Encoder</h4>
        <p>17 Conformer blocks with multi-head self-attention, depthwise convolutions,
           and feed-forward layers downsample the features 8&times; in time.</p>
      </div>
      <div class="app-explainer-phase">
        <h4>3. Transducer Decoder</h4>
        <p>An RNNT (or TDT) decoder with an LSTM prediction network emits subword
           tokens frame-by-frame. TDT skips frames for 2&ndash;5&times; faster decoding.</p>
      </div>
      <div class="app-explainer-phase">
        <h4>4. SentencePiece Detokenizer</h4>
        <p>Subword token IDs are mapped back to text using the model's BPE vocabulary.</p>
      </div>
    </div>
    <p class="app-explainer-detail">
      Model weights are cached in the browser after the first download using the
      <code>Cache API</code>, so subsequent loads are instant.
    </p>
  `;
  wrapper.appendChild(explainer);

  container.appendChild(wrapper);

  // ---- wiring ----

  loadBtn.addEventListener('click', async () => {
    if (recognizer) return;
    loadBtn.disabled = true;
    loadBtn.textContent = 'Loading...';

    try {
      const spec = MODELS[modelSelect.value];
      const backend = backendSelect.value as 'webgpu' | 'webgl';

      setStatus('Downloading config & vocab...');
      setProgress(0);

      const [configJson, vocabJson] = await Promise.all([
        fetchTextCached(spec.configUrl),
        fetchTextCached(spec.vocabUrl),
      ]);

      setStatus(`Downloading model weights (~${spec.sizeMB} MB)...`);
      const modelBuf = await fetchCached(spec.weightsUrl, pct => setProgress(pct * 0.9));

      if (destroyed) return;

      setStatus('Initialising TensorFlow.js...');
      setProgress(0.92);

      recognizer = new FastConformerASR({
        sampleRate: INPUT_SAMPLE_RATE,
        modelPath: spec.weightsUrl,
        configPath: spec.configUrl,
        vocabPath: spec.vocabUrl,
        backend,
        inputSampleRate: INPUT_SAMPLE_RATE,
        streaming: true,
        chunkSizeMs: 2000,
      });

      await recognizer.loadFromBuffers(modelBuf, configJson, vocabJson);

      if (destroyed) return;

      setProgress(1);
      setStatus(`Model loaded (${backend.toUpperCase()})`);
      showAudioUI();
    } catch (err) {
      setStatus(`Error: ${err instanceof Error ? err.message : String(err)}`);
      loadBtn.disabled = false;
      loadBtn.textContent = 'Retry';
    }
  });

  function showAudioUI() {
    audioSection.style.display = '';
    transcriptSection.style.display = '';
    loadBtn.style.display = 'none';
    modelSelect.style.display = 'none';
    backendSelect.style.display = 'none';
    backendHint.style.display = 'none';

    audioInput = new AudioInput(INPUT_SAMPLE_RATE);
    new AudioInputUI(audioSection, audioInput);

    // VAD endpointer detects pauses and triggers offline transcription
    // of each utterance segment for accurate results while recording.
    const endpointer = new Endpointer({
      sampleRate: INPUT_SAMPLE_RATE,
      silenceTimeoutMs: 1200,
    });

    let utteranceChunks: Float32Array[] = [];
    let allChunks: Float32Array[] = [];
    let transcribing = false;

    async function transcribeUtterance() {
      if (!recognizer || utteranceChunks.length === 0 || transcribing) return;
      transcribing = true;

      const totalLen = utteranceChunks.reduce((s, c) => s + c.length, 0);
      const audio = new Float32Array(totalLen);
      let off = 0;
      for (const c of utteranceChunks) { audio.set(c, off); off += c.length; }
      utteranceChunks = [];

      const audioDurationS = totalLen / INPUT_SAMPLE_RATE;
      if (audioDurationS < 0.3) { transcribing = false; return; }

      partialLine.textContent = `Transcribing ${audioDurationS.toFixed(1)}s\u2026`;
      const result = await recognizer.transcribe(audio);
      if (result.text.trim()) {
        appendTranscript(result, audioDurationS);
      }
      partialLine.textContent = '';
      recognizer.reset();
      transcribing = false;
    }

    audioInput.on('pcm-data', async (pcm: Float32Array) => {
      if (!recognizer) return;
      const copy = new Float32Array(pcm);
      utteranceChunks.push(copy);
      allChunks.push(copy);

      // VAD-based utterance segmentation
      const event = endpointer.processFrame(pcm);
      if (event === 'speech') {
        const dur = utteranceChunks.reduce((s, c) => s + c.length, 0) / INPUT_SAMPLE_RATE;
        partialLine.textContent = `Recording\u2026 (${dur.toFixed(1)}s)`;
      } else if (event === 'speech-end') {
        transcribeUtterance();
      }

      // Also feed streaming for live preview text
      const result = await recognizer.processFrameAsync(pcm);
      if (result && result.text.trim()) {
        latencyLabel.textContent = `Preview: ${result.text}`;
      }
    });

    audioInput.on('start', () => {
      utteranceChunks = [];
      allChunks = [];
      endpointer.reset();
      partialLine.textContent = '(listening\u2026)';
      latencyLabel.textContent = '';
      recognizer?.start();
    });

    audioInput.on('stop', async () => {
      if (!recognizer) return;

      // Transcribe any remaining audio in the current utterance
      if (utteranceChunks.length > 0) {
        await transcribeUtterance();
      }

      partialLine.textContent = '';
      latencyLabel.textContent = '';
      recognizer.reset();
    });
  }

  function appendTranscript(result: ASRResult, audioDurationS?: number) {
    const entry = el('div', 'asr-transcript-entry');
    const time = new Date().toLocaleTimeString();
    const inferMs = Math.round(result.latencyMs);
    let meta = `${inferMs} ms · ${result.decoderType.toUpperCase()} · ${result.tokenCount} tokens`;
    if (audioDurationS && audioDurationS > 0) {
      const rtf = (result.latencyMs / 1000 / audioDurationS).toFixed(2);
      meta += ` · ${audioDurationS.toFixed(1)}s audio · RTF ${rtf}x`;
    }
    entry.innerHTML = `<span class="asr-transcript-time">[${time}]</span> ${escapeHtml(result.text)} <span class="asr-transcript-meta">${meta}</span>`;
    transcriptLog.prepend(entry);
  }

  // ---- helpers ----

  function setProgress(pct: number) {
    progressFill.style.width = `${Math.round(pct * 100)}%`;
  }

  function setStatus(msg: string) {
    statusLabel.textContent = msg;
  }

  // ---- cleanup ----
  return () => {
    destroyed = true;
    audioInput?.stop();
    recognizer?.stop();
    wrapper.remove();
  };
}

/* ------------------------------------------------------------------ */
/*  Tiny DOM helpers                                                   */
/* ------------------------------------------------------------------ */

function el(tag: string, className?: string): HTMLDivElement {
  const e = document.createElement(tag) as HTMLDivElement;
  if (className) e.className = className;
  return e;
}

function escapeHtml(s: string): string {
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}
