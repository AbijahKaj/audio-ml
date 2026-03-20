import { SpeechRecognizer } from 'audio-ml/applications';

import { AudioInput } from '../components/AudioInput';
import { AudioInputUI } from '../components/AudioInputUI';

const sampleRate = 44100;
const sampleAudioUrl = '/jfk.wav';
const sampleExpectation = 'ask not what your country can do for you';

function normalizeText(text: string): string {
  return text.toLowerCase().replace(/[^a-z0-9\s]/g, ' ').replace(/\s+/g, ' ').trim();
}

async function decodeAudioFromUrl(url: string): Promise<Float32Array> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch sample audio: ${response.status} ${response.statusText}`);
  }

  const buffer = await response.arrayBuffer();
  const audioContext = new AudioContext();
  try {
    const decoded = await audioContext.decodeAudioData(buffer.slice(0));
    const output = new Float32Array(decoded.length);
    for (let channel = 0; channel < decoded.numberOfChannels; channel += 1) {
      const channelData = decoded.getChannelData(channel);
      for (let index = 0; index < decoded.length; index += 1) {
        output[index] += channelData[index] / decoded.numberOfChannels;
      }
    }
    return output;
  } finally {
    await audioContext.close();
  }
}

export function createSpeechRecognizerDemo(container: HTMLElement): () => void {
  const root = document.createElement('div');
  root.className = 'speech-recognizer-demo';
  root.style.cssText = 'display: flex; flex-direction: column; gap: 1rem;';
  container.appendChild(root);

  const heading = document.createElement('h2');
  heading.textContent = 'Speech Recognizer';
  root.appendChild(heading);

  const description = document.createElement('p');
  description.textContent = 'This demo uses a real browser-capable ASR model through Transformers.js (default: onnx-community/whisper-tiny.en in the browser). Use the built-in sample or stream microphone/file audio for partial and final transcripts.';
  root.appendChild(description);

  const controls = document.createElement('div');
  controls.style.cssText = 'display: flex; gap: 0.75rem; flex-wrap: wrap; align-items: center;';
  root.appendChild(controls);

  const loadButton = document.createElement('button');
  loadButton.textContent = 'Load ASR Model';
  controls.appendChild(loadButton);

  const sampleButton = document.createElement('button');
  sampleButton.textContent = 'Transcribe Built-in Sample';
  controls.appendChild(sampleButton);

  const status = document.createElement('div');
  status.id = 'speech-recognizer-status';
  status.style.cssText = 'padding: 0.75rem; border: 1px solid #333; border-radius: 8px; background: #111;';
  status.textContent = 'Model not loaded';
  root.appendChild(status);

  const transcriptCard = document.createElement('div');
  transcriptCard.style.cssText = 'padding: 1rem; border: 1px solid #333; border-radius: 8px; background: #111;';
  root.appendChild(transcriptCard);

  const partialTitle = document.createElement('h3');
  partialTitle.textContent = 'Partial Transcript';
  transcriptCard.appendChild(partialTitle);

  const partialText = document.createElement('div');
  partialText.id = 'speech-recognizer-partial';
  partialText.textContent = '—';
  partialText.style.cssText = 'min-height: 1.5rem; color: #9cc6ff;';
  transcriptCard.appendChild(partialText);

  const finalTitle = document.createElement('h3');
  finalTitle.textContent = 'Final Transcript';
  transcriptCard.appendChild(finalTitle);

  const finalText = document.createElement('div');
  finalText.id = 'speech-recognizer-final';
  finalText.textContent = '—';
  finalText.style.cssText = 'min-height: 2rem; color: #d6ffd6; font-weight: 600;';
  transcriptCard.appendChild(finalText);

  const metrics = document.createElement('div');
  metrics.id = 'speech-recognizer-metrics';
  metrics.textContent = 'Latency: n/a';
  metrics.style.cssText = 'color: #aaa; margin-top: 0.75rem;';
  transcriptCard.appendChild(metrics);

  const logTitle = document.createElement('h3');
  logTitle.textContent = 'Event Log';
  root.appendChild(logTitle);

  const log = document.createElement('div');
  log.id = 'speech-recognizer-log';
  log.style.cssText = 'max-height: 220px; overflow: auto; padding: 0.75rem; border: 1px solid #333; border-radius: 8px; background: #111; display: flex; flex-direction: column; gap: 0.5rem;';
  root.appendChild(log);

  const audioInput = new AudioInput(sampleRate);
  new AudioInputUI(root, audioInput);

  const recognizer = new SpeechRecognizer({
    provider: 'transformers-js',
    modelId: 'onnx-community/whisper-tiny.en',
    sampleRate,
    transformersDtype: 'q8',
    chunkDurationMs: 800,
    partialDecodeEveryChunks: 2,
    chunkLengthSeconds: 10,
    strideLengthSeconds: 2,
    vadConfig: {
      fftSize: 1024,
      silenceFramesRequired: 3,
      speechFramesRequired: 2,
    },
  });

  let isLoaded = false;

  const addLog = (message: string): void => {
    const entry = document.createElement('div');
    entry.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
    log.appendChild(entry);
    log.scrollTop = log.scrollHeight;
  };

  const updateStatus = (message: string): void => {
    status.textContent = message;
  };

  const ensureLoaded = async (): Promise<void> => {
    if (isLoaded) {
      return;
    }

    updateStatus('Loading model...');
    loadButton.disabled = true;
    sampleButton.disabled = true;
    await recognizer.load();
    isLoaded = true;
    loadButton.textContent = 'Model Loaded';
    updateStatus('Model ready');
    sampleButton.disabled = false;
  };

  const runSample = async (): Promise<void> => {
    await ensureLoaded();
    updateStatus('Downloading and transcribing sample...');
    const audio = await decodeAudioFromUrl(sampleAudioUrl);
    const result = await recognizer.transcribe(audio);
    finalText.textContent = result.text || '—';
    metrics.textContent = `Latency: ${Math.round(result.latencyMs)} ms`;
    addLog(`Sample transcription complete: "${result.text}"`);
    updateStatus('Sample transcription complete');

    const pass = normalizeText(result.text).includes(sampleExpectation);
    document.body.dataset.asrAutotest = pass ? 'pass' : 'fail';
    document.body.dataset.asrTranscript = result.text;
  };

  loadButton.addEventListener('click', () => {
    void ensureLoaded().catch((error: unknown) => {
      const message = error instanceof Error ? error.message : String(error);
      updateStatus(`Load failed: ${message}`);
      addLog(`Load failed: ${message}`);
    });
  });

  sampleButton.addEventListener('click', () => {
    void runSample().catch((error: unknown) => {
      const message = error instanceof Error ? error.message : String(error);
      updateStatus(`Sample transcription failed: ${message}`);
      addLog(`Sample transcription failed: ${message}`);
      document.body.dataset.asrAutotest = 'fail';
    });
  });

  recognizer.on('progress', (info: unknown) => {
    const maybeInfo = info as { status?: string; file?: string; progress?: number };
    const progress = typeof maybeInfo.progress === 'number' ? ` ${(maybeInfo.progress * 100).toFixed(0)}%` : '';
    updateStatus(`Loading: ${maybeInfo.status ?? 'working'}${progress}${maybeInfo.file ? ` (${maybeInfo.file})` : ''}`);
  });

  recognizer.on('ready', (info: { provider?: string; modelId?: string }) => {
    addLog(`Recognizer ready (${info.provider ?? 'unknown'}${info.modelId ? ` / ${info.modelId}` : ''})`);
    updateStatus('Model ready');
  });

  recognizer.on('partial', (result: { text: string; latencyMs: number }) => {
    partialText.textContent = result.text || '—';
    metrics.textContent = `Latency: ${Math.round(result.latencyMs)} ms`;
  });

  recognizer.on('final', (result: { text: string; latencyMs: number }) => {
    finalText.textContent = result.text || '—';
    metrics.textContent = `Latency: ${Math.round(result.latencyMs)} ms`;
    addLog(`Final transcript: "${result.text}"`);
  });

  recognizer.on('error', (error: unknown) => {
    const message = error instanceof Error ? error.message : String(error);
    updateStatus(`Recognizer error: ${message}`);
    addLog(`Recognizer error: ${message}`);
  });

  const pcmHandler = (pcm: Float32Array) => {
    if (isLoaded) {
      recognizer.processFrame(pcm);
    }
  };

  audioInput.on('pcm-data', pcmHandler);
  audioInput.on('start', () => {
    if (!isLoaded) {
      void ensureLoaded().catch((error: unknown) => {
        const message = error instanceof Error ? error.message : String(error);
        updateStatus(`Load failed: ${message}`);
        addLog(`Load failed: ${message}`);
      });
      return;
    }

    recognizer.start();
    partialText.textContent = 'Listening...';
    addLog('Streaming transcription started');
  });

  audioInput.on('stop', () => {
    void recognizer.finalize().catch((error: unknown) => {
      const message = error instanceof Error ? error.message : String(error);
      updateStatus(`Finalize failed: ${message}`);
      addLog(`Finalize failed: ${message}`);
    });
    recognizer.stop();
    addLog('Streaming transcription stopped');
  });

  const params = new URLSearchParams(window.location.search);
  if (params.get('autotest') === '1') {
    void runSample().catch((error: unknown) => {
      document.body.dataset.asrAutotest = 'fail';
      const message = error instanceof Error ? error.message : String(error);
      updateStatus(`Autotest failed: ${message}`);
      addLog(`Autotest failed: ${message}`);
    });
  }

  return () => {
    audioInput.off('pcm-data', pcmHandler);
    void audioInput.stop();
    recognizer.reset();
    root.remove();
  };
}
