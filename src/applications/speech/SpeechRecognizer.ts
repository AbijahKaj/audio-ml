import { BaseApplication, type ApplicationConfig } from '../base/BaseApplication';
import { TfjsBackend } from '../../asr/compute/TfjsBackend';
import type { TfjsBackendName } from '../../asr/compute/TfjsBackend';
import { loadSafeTensorsSync } from '../../asr/model/SafeTensorsLoader';
import { parseModelConfig, type FastConformerConfig } from '../../asr/model/ModelConfig';
import { mapWeights } from '../../asr/model/WeightMapper';
import { FeaturePipeline } from '../../asr/features/FeaturePipeline';
import { Resampler } from '../../asr/features/Resampler';
import { FastConformerEncoder } from '../../asr/encoder/FastConformerEncoder';
import { createDecoder } from '../../asr/decoder/createDecoder';
import type { RNNTGreedyDecoder } from '../../asr/decoder/RNNTGreedyDecoder';
import type { TDTGreedyDecoder } from '../../asr/decoder/TDTGreedyDecoder';
import { SentencePieceDecoder } from '../../asr/text/SentencePieceDecoder';

export interface SpeechRecognizerConfig extends ApplicationConfig {
  modelUrl: string;
  configJson: string;
  vocabJson: string;
  backend?: TfjsBackendName;
  /** If input sample rate differs from model (16k), set this for resampling */
  inputSampleRate?: number;
  /** Enable streaming partials using buffered PCM + periodic decode */
  streaming?: boolean;
  /** Emit partial hypothesis every N ms when streaming (default 500) */
  partialIntervalMs?: number;
}

export interface ASRResult {
  text: string;
  isFinal: boolean;
  latencyMs: number;
  decoderType: 'rnnt' | 'tdt';
}

async function fetchText(url: string): Promise<string> {
  const r = await fetch(url);
  if (!r.ok) {
    throw new Error(`Failed to fetch ${url}: ${r.status}`);
  }
  return r.text();
}

async function fetchArrayBuffer(url: string): Promise<ArrayBuffer> {
  const r = await fetch(url);
  if (!r.ok) {
    throw new Error(`Failed to fetch ${url}: ${r.status}`);
  }
  return r.arrayBuffer();
}

/**
 * FastConformer RNNT/TDT speech recognizer (TensorFlow.js backend).
 */
export class SpeechRecognizer extends BaseApplication {
  private backendImpl: InstanceType<typeof TfjsBackend> | null = null;
  private config: FastConformerConfig | null = null;
  private featurePipeline: FeaturePipeline | null = null;
  private encoder: FastConformerEncoder | null = null;
  private decoder: RNNTGreedyDecoder | TDTGreedyDecoder | null = null;
  private tokenizer: SentencePieceDecoder | null = null;
  private resampler: Resampler | null = null;

  private pcmBuffer: Float32Array[] = [];
  private streamTimer: ReturnType<typeof setInterval> | null = null;
  private loaded = false;

  private readonly modelUrl: string;
  private readonly configUrl: string;
  private readonly vocabUrl: string;
  private readonly backendPref: TfjsBackendName;
  private readonly inputSr: number;
  private readonly useStreaming: boolean;
  private readonly partialMs: number;

  constructor(cfg: SpeechRecognizerConfig) {
    super(cfg);
    this.modelUrl = cfg.modelUrl;
    this.configUrl = cfg.configJson;
    this.vocabUrl = cfg.vocabJson;
    this.backendPref = cfg.backend ?? 'wasm';
    this.inputSr = cfg.inputSampleRate ?? cfg.sampleRate;
    this.useStreaming = cfg.streaming ?? false;
    this.partialMs = cfg.partialIntervalMs ?? 500;
  }

  async load(): Promise<void> {
    await TfjsBackend.init(this.backendPref);
    this.backendImpl = new TfjsBackend();

    const configJson = await fetchText(this.configUrl);
    this.config = parseModelConfig(configJson);

    const buf = await fetchArrayBuffer(this.modelUrl);
    const raw = loadSafeTensorsSync(buf, this.backendImpl!);
    const weights = mapWeights(raw, this.config, this.backendImpl);

    this.featurePipeline = new FeaturePipeline(this.config);
    this.encoder = new FastConformerEncoder(this.backendImpl, weights.encoder, this.config);
    this.decoder = createDecoder(this.config, this.backendImpl, weights.decoder);
    this.tokenizer = new SentencePieceDecoder(await fetchText(this.vocabUrl));

    if (this.inputSr !== this.config.sampleRate) {
      this.resampler = new Resampler(this.inputSr, this.config.sampleRate);
    }

    this.loaded = true;
    this.emit('ready', { decoderType: this.config.decoderType });
  }

  processFrame(pcm: Float32Array): void {
    if (!this.useStreaming || !this.loaded) {
      return;
    }
    this.pcmBuffer.push(pcm.slice());
    if (!this.streamTimer) {
      this.streamTimer = setInterval(() => void this.emitPartial(), this.partialMs);
    }
  }

  private async emitPartial(): Promise<void> {
    if (!this.backendImpl || !this.config || !this.featurePipeline || !this.encoder || !this.decoder || !this.tokenizer) {
      return;
    }
    if (this.pcmBuffer.length === 0) {
      return;
    }
    const total = this.pcmBuffer.reduce((n, f) => n + f.length, 0);
    const cat = new Float32Array(total);
    let o = 0;
    for (const f of this.pcmBuffer) {
      cat.set(f, o);
      o += f.length;
    }
    let audio: Float32Array = cat;
    if (this.resampler) {
      audio = Float32Array.from(this.resampler.resample(cat));
    }
    const mel = this.featurePipeline.extractFeatures(this.backendImpl, audio);
    const enc = this.encoder.forward(mel);
    this.backendImpl.dispose(mel);
    const ids = await this.decoder.decode(enc);
    this.backendImpl.dispose(enc);
    const text = this.tokenizer.decode(ids);
    this.emit('partial', { text, isFinal: false });
  }

  async transcribe(audio: Float32Array): Promise<ASRResult> {
    if (!this.backendImpl || !this.config || !this.featurePipeline || !this.encoder || !this.decoder || !this.tokenizer) {
      throw new Error('SpeechRecognizer.load() must complete before transcribe()');
    }
    const t0 = typeof performance !== 'undefined' ? performance.now() : Date.now();
    let pcm: Float32Array = audio;
    if (this.resampler) {
      pcm = Float32Array.from(this.resampler.resample(audio));
    }
    const mel = this.featurePipeline.extractFeatures(this.backendImpl, pcm);
    const enc = this.encoder.forward(mel);
    this.backendImpl.dispose(mel);
    const ids = await this.decoder.decode(enc);
    this.backendImpl.dispose(enc);
    const text = this.tokenizer.decode(ids);
    const t1 = typeof performance !== 'undefined' ? performance.now() : Date.now();
    return {
      text,
      isFinal: true,
      latencyMs: t1 - t0,
      decoderType: this.config.decoderType,
    };
  }

  reset(): void {
    super.reset();
    this.pcmBuffer = [];
    if (this.streamTimer) {
      clearInterval(this.streamTimer);
      this.streamTimer = null;
    }
  }
}
