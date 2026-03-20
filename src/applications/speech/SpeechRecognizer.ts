import type { ComputeBackend } from '../../asr/compute/Backend';
import { TfjsBackend, type TfjsBackendName } from '../../asr/compute/TfjsBackend';
import { createDecoder } from '../../asr/decoder/createDecoder';
import type { RNNTGreedyDecoder } from '../../asr/decoder/RNNTGreedyDecoder';
import type { TDTGreedyDecoder } from '../../asr/decoder/TDTGreedyDecoder';
import { FeaturePipeline } from '../../asr/features/FeaturePipeline';
import { Resampler } from '../../asr/features/Resampler';
import { FastConformerEncoder } from '../../asr/encoder/FastConformerEncoder';
import { loadSafeTensors } from '../../asr/model/SafeTensorsLoader';
import { mapWeights } from '../../asr/model/WeightMapper';
import { parseModelConfig, type FastConformerConfig } from '../../asr/model/ModelConfig';
import { SentencePieceDecoder } from '../../asr/text/SentencePieceDecoder';
import { ChunkedInference, type StreamingAsrHost } from '../../asr/streaming/ChunkedInference';
import { BaseApplication, type ApplicationConfig } from '../base/BaseApplication';

export interface SpeechRecognizerConfig extends ApplicationConfig {
  modelPath: string;
  configPath: string;
  vocabPath: string;
  backend?: TfjsBackendName;
  /** Override sample rate for mic (resample to model rate). */
  inputSampleRate?: number;
}

export interface ASRResult {
  text: string;
  isFinal: boolean;
  latencyMs: number;
  decoderType: 'rnnt' | 'tdt';
}

async function readText(source: string): Promise<string> {
  if (source.startsWith('data:')) {
    return decodeURIComponent(source.split(',')[1] ?? '');
  }
  const res = await fetch(source);
  return res.text();
}

async function readArrayBuffer(source: string): Promise<ArrayBuffer> {
  const res = await fetch(source);
  return res.arrayBuffer();
}

export class SpeechRecognizer extends BaseApplication {
  private backend: ComputeBackend | null = null;
  private config: FastConformerConfig | null = null;
  private featurePipeline: FeaturePipeline | null = null;
  private encoder: FastConformerEncoder | null = null;
  private decoder: RNNTGreedyDecoder | TDTGreedyDecoder | null = null;
  private tokenizer: SentencePieceDecoder | null = null;
  private resampler: Resampler | null = null;
  private readonly modelPath: string;
  private readonly configPath: string;
  private readonly vocabPath: string;
  private readonly backendName: TfjsBackendName;
  private readonly inputSampleRate: number;
  private isLoaded = false;
  private streaming: ChunkedInference | null = null;

  constructor(cfg: SpeechRecognizerConfig) {
    super(cfg);
    this.modelPath = cfg.modelPath;
    this.configPath = cfg.configPath;
    this.vocabPath = cfg.vocabPath;
    this.backendName = cfg.backend ?? 'cpu';
    this.inputSampleRate = cfg.inputSampleRate ?? cfg.sampleRate;
  }

  async load(): Promise<void> {
    const backend = new TfjsBackend(this.backendName);
    const configJson = await readText(this.configPath);
    const config = parseModelConfig(configJson);
    const buf = await readArrayBuffer(this.modelPath);
    const rawWeights = await loadSafeTensors(buf, backend);
    const modelWeights = mapWeights(rawWeights, config, backend);
    const vocabJson = await readText(this.vocabPath);

    this.backend = backend;
    this.config = config;
    this.featurePipeline = new FeaturePipeline(config);
    this.encoder = new FastConformerEncoder(backend, modelWeights.encoder, config);
    this.decoder = createDecoder(config, backend, modelWeights.decoder);
    this.tokenizer = new SentencePieceDecoder(vocabJson);
    this.resampler = new Resampler(this.inputSampleRate, config.sampleRate);
    this.streaming = new ChunkedInference(this as unknown as StreamingAsrHost);
    this.isLoaded = true;
    this.emit('ready', { decoderType: config.decoderType });
  }

  getDecoderType(): 'rnnt' | 'tdt' | null {
    return this.config?.decoderType ?? null;
  }

  processFrame(pcm: Float32Array): void {
    if (!this.isLoaded || !this.streaming) {
      return;
    }
    this.streaming.pushPcm(pcm);
  }

  /** Call after VAD `speech-end` (or manual endpoint) to emit `final` and clear the stream buffer. */
  async finalizeStream(): Promise<void> {
    await this.streaming?.finalize();
  }

  async transcribe(audio: Float32Array): Promise<ASRResult> {
    const start = typeof performance !== 'undefined' ? performance.now() : Date.now();
    if (!this.backend || !this.config || !this.featurePipeline || !this.encoder || !this.decoder || !this.tokenizer) {
      throw new Error('SpeechRecognizer.load() must be called first');
    }
    const backend = this.backend;
    let pcm = audio;
    if (this.resampler && this.inputSampleRate !== this.config.sampleRate) {
      pcm = this.resampler.resample(audio);
    }
    const mel = this.featurePipeline.extractFeatures(pcm, backend);
    const encoded = this.encoder.forward(mel);
    const tokenIds = await this.decoder.decode(encoded);
    const text = this.tokenizer.decode(tokenIds);
    backend.dispose(mel);
    backend.dispose(encoded);

    const end = typeof performance !== 'undefined' ? performance.now() : Date.now();
    return {
      text,
      isFinal: true,
      latencyMs: end - start,
      decoderType: this.config.decoderType,
    };
  }

  /** Internal: run ASR on a PCM buffer (used by ChunkedInference). */
  async transcribeBuffer(buffer: Float32Array): Promise<ASRResult> {
    return this.transcribe(buffer);
  }

  reset(): void {
    super.reset();
    this.streaming?.reset();
  }
}
