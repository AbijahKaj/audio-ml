/**
 * Speech Recognizer Demo Page
 *
 * Demonstrates FastConformer RNNT/TDT ASR with streaming microphone input.
 * Requires model files (parakeet_120m.safetensors, model_config.json, vocab.json)
 * to be available at the configured paths, or uploaded via the file inputs.
 */

import { SpeechRecognizer } from 'audio-ml/applications';
import { AudioInput } from '../components/AudioInput';
import { AudioInputUI } from '../components/AudioInputUI';

const MODEL_BACKEND = (
  typeof (navigator as Navigator & { gpu?: unknown }).gpu !== 'undefined'
    ? 'webgpu'
    : 'wasm'
) as 'webgpu' | 'wasm';

export function createSpeechRecognizerDemo(container: HTMLElement): () => void {
  let recognizer: SpeechRecognizer | null = null;
  let audioInput: AudioInput | null = null;
  let isLoading = false;

  // ── Layout ──────────────────────────────────────────────────────────────────

  const header = document.createElement('div');
  header.className = 'asr-header';
  header.innerHTML = `
    <h2 class="asr-title">FastConformer ASR</h2>
    <p class="asr-subtitle">
      Real-time speech recognition using NVIDIA NeMo's FastConformer architecture
      with RNNT and TDT decoder support. Runs entirely in the browser via TensorFlow.js.
    </p>
  `;
  container.appendChild(header);

  // Model loading section
  const loadSection = document.createElement('div');
  loadSection.className = 'asr-load-section';
  loadSection.innerHTML = `
    <h3>Load Model</h3>
    <p class="asr-load-hint">
      Export a NeMo model using <code>tools/export_nemo_to_safetensors.py</code>, then upload the
      three output files below. Or point to URLs if served from a CDN.
    </p>
    <div class="asr-file-inputs">
      <div class="asr-file-input-group">
        <label>Model weights (.safetensors)</label>
        <input type="file" id="asr-model-file" accept=".safetensors" />
      </div>
      <div class="asr-file-input-group">
        <label>Config (model_config.json)</label>
        <input type="file" id="asr-config-file" accept=".json" />
      </div>
      <div class="asr-file-input-group">
        <label>Vocabulary (vocab.json)</label>
        <input type="file" id="asr-vocab-file" accept=".json" />
      </div>
    </div>
    <div class="asr-backend-selector">
      <label>Compute backend:</label>
      <select id="asr-backend-select">
        <option value="wasm">WASM (CPU — universal)</option>
        <option value="webgpu" ${MODEL_BACKEND === 'webgpu' ? 'selected' : ''}>WebGPU (GPU — faster)</option>
        <option value="webgl">WebGL (GPU — fallback)</option>
        <option value="cpu">CPU (debug)</option>
      </select>
    </div>
    <div class="asr-load-actions">
      <button id="asr-load-btn" class="asr-btn asr-btn-primary" disabled>Load Model</button>
      <span id="asr-load-status" class="asr-load-status"></span>
    </div>
    <div id="asr-load-progress" class="asr-progress" style="display:none">
      <div class="asr-progress-bar" id="asr-progress-bar"></div>
    </div>
  `;
  container.appendChild(loadSection);

  // Audio input
  const audioSection = document.createElement('div');
  audioSection.className = 'asr-audio-section';
  audioInput = new AudioInput(16000);
  new AudioInputUI(audioSection, audioInput);
  container.appendChild(audioSection);

  // Transcription display
  const transcriptSection = document.createElement('div');
  transcriptSection.className = 'asr-transcript-section';
  transcriptSection.innerHTML = `
    <h3>Transcription</h3>
    <div class="asr-transcript-display">
      <div id="asr-final-text" class="asr-final-text"></div>
      <span id="asr-partial-text" class="asr-partial-text"></span>
    </div>
    <div class="asr-transcript-controls">
      <button id="asr-clear-btn" class="asr-btn">Clear</button>
      <span id="asr-latency" class="asr-latency"></span>
      <span id="asr-decoder-type" class="asr-decoder-badge"></span>
    </div>
  `;
  container.appendChild(transcriptSection);

  // Status bar
  const statusBar = document.createElement('div');
  statusBar.id = 'asr-status-bar';
  statusBar.className = 'asr-status-bar';
  statusBar.textContent = 'Load a model to begin.';
  container.appendChild(statusBar);

  // Explainer
  const explainer = document.createElement('div');
  explainer.className = 'app-explainer';
  explainer.innerHTML = `
    <h2 class="app-explainer-title">How It Works</h2>
    <p class="app-explainer-description">
      This demo runs a full FastConformer ASR pipeline in the browser:
    </p>
    <ol class="asr-explainer-steps">
      <li><strong>Feature Extraction</strong> — Microphone audio is resampled to 16 kHz and
          converted to 80-band log-mel spectrogram features (25 ms window, 10 ms hop).</li>
      <li><strong>FastConformer Encoder</strong> — 17–24 Conformer blocks (Conv subsampling + 
          multi-head self-attention with relative positional encoding + depthwise conv + 
          Macaron FFN sandwich) reduce time by 8× and produce rich acoustic representations.</li>
      <li><strong>RNNT or TDT Decoder</strong> — A prediction network (LSTM) and joint network
          greedily decode token IDs from the encoder output. TDT is 2–5× faster by 
          predicting how many frames to skip.</li>
      <li><strong>SentencePiece Detokenization</strong> — Token IDs are converted to text
          using the model's BPE vocabulary.</li>
    </ol>
    <p class="app-explainer-detail">
      Supports all NeMo FastConformer-RNNT and FastConformer-TDT checkpoints exported via
      <code>tools/export_nemo_to_safetensors.py</code>, including multilingual models
      (Parakeet TDT 0.6B v3 — English + French + 23 other languages).
    </p>
  `;
  container.appendChild(explainer);

  // ── DOM refs ─────────────────────────────────────────────────────────────────

  const loadBtn = container.querySelector<HTMLButtonElement>('#asr-load-btn')!;
  const loadStatus = container.querySelector<HTMLSpanElement>('#asr-load-status')!;
  const modelFileInput = container.querySelector<HTMLInputElement>('#asr-model-file')!;
  const configFileInput = container.querySelector<HTMLInputElement>('#asr-config-file')!;
  const vocabFileInput = container.querySelector<HTMLInputElement>('#asr-vocab-file')!;
  const backendSelect = container.querySelector<HTMLSelectElement>('#asr-backend-select')!;
  const progressDiv = container.querySelector<HTMLDivElement>('#asr-load-progress')!;
  const finalTextDiv = container.querySelector<HTMLDivElement>('#asr-final-text')!;
  const partialTextSpan = container.querySelector<HTMLSpanElement>('#asr-partial-text')!;
  const latencySpan = container.querySelector<HTMLSpanElement>('#asr-latency')!;
  const decoderBadge = container.querySelector<HTMLSpanElement>('#asr-decoder-type')!;
  const statusBarEl = container.querySelector<HTMLDivElement>('#asr-status-bar')!;
  const clearBtn = container.querySelector<HTMLButtonElement>('#asr-clear-btn')!;

  // ── File input handling ───────────────────────────────────────────────────────

  function checkFilesReady(): void {
    loadBtn.disabled = !(modelFileInput.files?.length && configFileInput.files?.length && vocabFileInput.files?.length);
  }

  modelFileInput.addEventListener('change', checkFilesReady);
  configFileInput.addEventListener('change', checkFilesReady);
  vocabFileInput.addEventListener('change', checkFilesReady);

  // ── Load model ────────────────────────────────────────────────────────────────

  loadBtn.addEventListener('click', async () => {
    if (isLoading || !modelFileInput.files?.length) return;
    isLoading = true;
    loadBtn.disabled = true;
    loadStatus.textContent = 'Loading…';
    progressDiv.style.display = 'block';

    try {
      const modelBuffer = await readFileAsArrayBuffer(modelFileInput.files[0]);
      const configText = await readFileAsText(configFileInput.files![0]);
      const vocabText = await readFileAsText(vocabFileInput.files![0]);

      // Write to object URLs so SpeechRecognizer can fetch them
      const configBlob = new Blob([configText], { type: 'application/json' });
      const vocabBlob = new Blob([vocabText], { type: 'application/json' });
      const configUrl = URL.createObjectURL(configBlob);
      const vocabUrl = URL.createObjectURL(vocabBlob);

      recognizer = new SpeechRecognizer({
        sampleRate: 16000,
        modelPath: modelBuffer,
        configPath: configUrl,
        vocabPath: vocabUrl,
        backend: backendSelect.value as 'wasm' | 'webgpu' | 'webgl' | 'cpu',
      });

      recognizer.on('ready', ({ decoderType }: { decoderType: string }) => {
        loadStatus.textContent = `Ready (${decoderType.toUpperCase()})`;
        progressDiv.style.display = 'none';
        statusBarEl.textContent = `Model loaded. Decoder: ${decoderType.toUpperCase()}. Click Start to begin transcription.`;
        decoderBadge.textContent = decoderType.toUpperCase();
        decoderBadge.className = `asr-decoder-badge asr-decoder-${decoderType}`;
        URL.revokeObjectURL(configUrl);
        URL.revokeObjectURL(vocabUrl);
      });

      recognizer.on('partial', ({ text }: { text: string }) => {
        partialTextSpan.textContent = text;
      });

      recognizer.on('final', ({ text, latencyMs, decoderType }: { text: string; latencyMs: number; decoderType: string }) => {
        if (text.trim()) {
          const p = document.createElement('p');
          p.textContent = text;
          finalTextDiv.appendChild(p);
          finalTextDiv.scrollTop = finalTextDiv.scrollHeight;
        }
        partialTextSpan.textContent = '';
        latencySpan.textContent = `${Math.round(latencyMs)}ms`;
        decoderBadge.textContent = decoderType.toUpperCase();
      });

      recognizer.on('error', ({ error }: { error: Error }) => {
        loadStatus.textContent = `Error: ${error.message}`;
        statusBarEl.textContent = `Error: ${error.message}`;
        console.error('ASR error:', error);
      });

      await recognizer.load();

    } catch (err) {
      loadStatus.textContent = `Failed: ${(err as Error).message}`;
      progressDiv.style.display = 'none';
      console.error('Model load failed:', err);
    } finally {
      isLoading = false;
      loadBtn.disabled = false;
    }
  });

  // ── Audio → ASR ───────────────────────────────────────────────────────────────

  const pcmHandler = (pcm: Float32Array): void => {
    recognizer?.processFrame(pcm);
  };

  audioInput.on('pcm-data', pcmHandler);

  audioInput.on('start', () => {
    recognizer?.start();
    statusBarEl.textContent = 'Listening…';
  });

  audioInput.on('stop', () => {
    recognizer?.stop();
    statusBarEl.textContent = 'Stopped.';
  });

  // ── Controls ──────────────────────────────────────────────────────────────────

  clearBtn.addEventListener('click', () => {
    finalTextDiv.innerHTML = '';
    partialTextSpan.textContent = '';
    latencySpan.textContent = '';
    recognizer?.reset();
  });

  // ── Cleanup ───────────────────────────────────────────────────────────────────

  return () => {
    audioInput?.off('pcm-data', pcmHandler);
    audioInput?.stop();
    recognizer?.stop();
    recognizer?.reset();
    header.remove();
    loadSection.remove();
    audioSection.remove();
    transcriptSection.remove();
    statusBar.remove();
    explainer.remove();
  };
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function readFileAsArrayBuffer(file: File): Promise<ArrayBuffer> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as ArrayBuffer);
    reader.onerror = reject;
    reader.readAsArrayBuffer(file);
  });
}

function readFileAsText(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = reject;
    reader.readAsText(file);
  });
}
