import { BaseApplication, type ApplicationConfig } from '../base/BaseApplication';
import { TfjsBackend } from '../../asr/compute/TfjsBackend';
import type { ComputeBackend } from '../../asr/compute/Backend';
import { FeaturePipeline } from '../../asr/features/FeaturePipeline';
import { Resampler } from '../../asr/features/Resampler';
import { FastConformerEncoder } from '../../asr/encoder/FastConformerEncoder';
import { createDecoder, type GreedyDecoder } from '../../asr/decoder/createDecoder';
import { SentencePieceDecoder } from '../../asr/text/SentencePieceDecoder';
import { parseModelConfig, type FastConformerConfig } from '../../asr/model/ModelConfig';
import { loadSafeTensors } from '../../asr/model/SafeTensorsLoader';
import { mapWeights } from '../../asr/model/WeightMapper';

export interface SpeechRecognizerConfig extends ApplicationConfig {
  modelPath: string;
  configPath: string;
  vocabPath: string;
  backend?: 'wasm' | 'webgpu' | 'webgl' | 'cpu';
  inputSampleRate?: number;
}

export interface ASRResult {
  text: string;
  isFinal: boolean;
  latencyMs: number;
  decoderType: 'rnnt' | 'tdt';
  tokenCount: number;
}

async function fetchText(url: string): Promise<string> {
  const response = await fetch(url);
  return response.text();
}

export class SpeechRecognizer extends BaseApplication {
  private backend!: ComputeBackend;
  private featurePipeline!: FeaturePipeline;
  private encoder!: FastConformerEncoder;
  private decoder!: GreedyDecoder;
  private tokenizer!: SentencePieceDecoder;
  private resampler: Resampler | null = null;
  private config!: FastConformerConfig;
  private audioBuffer: Float32Array[] = [];
  private isLoaded: boolean = false;

  private modelPath: string;
  private configPath: string;
  private vocabPath: string;
  private backendType: 'wasm' | 'webgpu' | 'webgl' | 'cpu';
  private inputSampleRate: number;

  constructor(config: SpeechRecognizerConfig) {
    super(config);
    this.modelPath = config.modelPath;
    this.configPath = config.configPath;
    this.vocabPath = config.vocabPath;
    this.backendType = config.backend ?? 'cpu';
    this.inputSampleRate = config.inputSampleRate ?? config.sampleRate;
  }

  async load(): Promise<void> {
    const tfjsBackend = new TfjsBackend();
    await tfjsBackend.init(this.backendType);
    this.backend = tfjsBackend;

    const configJson = await fetchText(this.configPath);
    this.config = parseModelConfig(configJson);

    const weights = await loadSafeTensors(this.modelPath, this.backend);
    const modelWeights = mapWeights(weights, this.config);

    this.featurePipeline = new FeaturePipeline(this.config, this.backend);
    this.encoder = new FastConformerEncoder(this.backend, modelWeights.encoder, this.config);
    this.decoder = createDecoder(this.config, this.backend, modelWeights.decoder);

    const vocabJson = await fetchText(this.vocabPath);
    this.tokenizer = new SentencePieceDecoder(vocabJson);

    if (this.inputSampleRate !== this.config.sampleRate) {
      this.resampler = new Resampler(this.inputSampleRate, this.config.sampleRate);
    }

    this.isLoaded = true;
    this.emit('ready', { decoderType: this.config.decoderType });
  }

  async loadFromBuffers(
    modelBuffer: ArrayBuffer,
    configJson: string,
    vocabJson: string,
  ): Promise<void> {
    const tfjsBackend = new TfjsBackend();
    await tfjsBackend.init(this.backendType);
    this.backend = tfjsBackend;

    this.config = parseModelConfig(configJson);

    const weights = await loadSafeTensors(modelBuffer, this.backend);
    const modelWeights = mapWeights(weights, this.config);

    this.featurePipeline = new FeaturePipeline(this.config, this.backend);
    this.encoder = new FastConformerEncoder(this.backend, modelWeights.encoder, this.config);
    this.decoder = createDecoder(this.config, this.backend, modelWeights.decoder);
    this.tokenizer = new SentencePieceDecoder(vocabJson);

    if (this.inputSampleRate !== this.config.sampleRate) {
      this.resampler = new Resampler(this.inputSampleRate, this.config.sampleRate);
    }

    this.isLoaded = true;
    this.emit('ready', { decoderType: this.config.decoderType });
  }

  processFrame(pcm: Float32Array): ASRResult | null {
    if (!this.isLoaded) return null;
    this.audioBuffer.push(pcm);
    return null;
  }

  async transcribe(audio: Float32Array): Promise<ASRResult> {
    if (!this.isLoaded) {
      throw new Error('Model not loaded. Call load() first.');
    }

    const start = performance.now();

    let processedAudio = audio;
    if (this.resampler) {
      processedAudio = this.resampler.resample(audio);
    }

    const mel = this.featurePipeline.extractFeatures(processedAudio);
    const encoded = this.encoder.forward(mel);
    const tokenIds = await this.decoder.decode(encoded);
    const text = this.tokenizer.decode(tokenIds);

    this.backend.dispose(mel);
    this.backend.dispose(encoded);

    const result: ASRResult = {
      text,
      isFinal: true,
      latencyMs: performance.now() - start,
      decoderType: this.config.decoderType,
      tokenCount: tokenIds.length,
    };

    this.emit('transcription', result);
    return result;
  }

  async transcribeBuffer(): Promise<ASRResult> {
    if (this.audioBuffer.length === 0) {
      return {
        text: '',
        isFinal: true,
        latencyMs: 0,
        decoderType: this.config?.decoderType ?? 'rnnt',
        tokenCount: 0,
      };
    }

    const totalLength = this.audioBuffer.reduce((sum, buf) => sum + buf.length, 0);
    const combined = new Float32Array(totalLength);
    let offset = 0;
    for (const buf of this.audioBuffer) {
      combined.set(buf, offset);
      offset += buf.length;
    }

    return this.transcribe(combined);
  }

  getModelConfig(): FastConformerConfig | null {
    return this.config ?? null;
  }

  getIsLoaded(): boolean {
    return this.isLoaded;
  }

  reset(): void {
    super.reset();
    this.audioBuffer = [];
  }
}
