import { BaseApplication, type ApplicationConfig } from '../base/BaseApplication';
import { TfjsBackend } from '../../asr/compute/TfjsBackend';
import type { ComputeBackend } from '../../asr/compute/Backend';
import { loadSafeTensors } from '../../asr/model/SafeTensorsLoader';
import { mapWeights, summarizeModelWeights } from '../../asr/model/WeightMapper';
import { parseModelConfig, type DecoderType, type FastConformerConfig } from '../../asr/model/ModelConfig';
import { FeaturePipeline } from '../../asr/features/FeaturePipeline';
import { FastConformerEncoder } from '../../asr/encoder/FastConformerEncoder';
import { createDecoder } from '../../asr/decoder/createDecoder';
import { TransducerDecoder } from '../../asr/decoder/TransducerDecoder';
import { SentencePieceDecoder } from '../../asr/text/SentencePieceDecoder';
import { Resampler } from '../../asr/features/Resampler';
import { CacheManager } from '../../asr/streaming/CacheManager';
import { ChunkedInference } from '../../asr/streaming/ChunkedInference';
import { Endpointer } from '../../asr/streaming/Endpointer';

type BackendType = 'wasm' | 'webgpu' | 'webgl' | 'cpu';
type ModelSource = string | ArrayBuffer | Uint8Array;
type VocabSource = string | string[];

interface FetchResponseLike {
  text(): Promise<string>;
}

type FetchLike = (input: string) => Promise<FetchResponseLike>;

export interface SpeechRecognizerConfig extends ApplicationConfig {
  modelPath?: string;
  configPath?: string;
  vocabPath?: string;
  modelSource?: ModelSource;
  configSource?: string;
  vocabSource?: VocabSource;
  backend?: BackendType;
}

export interface ASRResult {
  text: string;
  isFinal: boolean;
  latencyMs: number;
  decoderType: DecoderType;
}

export interface StreamingASRResult extends ASRResult {
  tokenIds: number[];
}

function nowMs(): number {
  return Date.now();
}

function getGlobalFetch(): FetchLike | undefined {
  return (globalThis as unknown as { fetch?: FetchLike }).fetch;
}

async function resolveText(pathOrText: string | undefined): Promise<string> {
  if (!pathOrText) {
    throw new Error('Missing text source.');
  }

  const trimmed = pathOrText.trimStart();
  if (trimmed.startsWith('{') || trimmed.startsWith('[')) {
    return pathOrText;
  }

  const fetchFn = getGlobalFetch();
  if (!fetchFn) {
    throw new Error(
      'String resource paths require global fetch(). Provide inline configSource/vocabSource in this environment.',
    );
  }

  const response = await fetchFn(pathOrText);
  return response.text();
}

export class SpeechRecognizer extends BaseApplication {
  private readonly backendType: BackendType;
  private backend: ComputeBackend | null = null;
  private modelConfig: FastConformerConfig | null = null;
  private featurePipeline: FeaturePipeline | null = null;
  private encoder: FastConformerEncoder | null = null;
  private decoder: TransducerDecoder | null = null;
  private tokenizer: SentencePieceDecoder | null = null;
  private resampler: Resampler | null = null;
  private chunkedInference: ChunkedInference | null = null;
  private tokenBuffer: number[] = [];
  private loaded = false;

  constructor(private config: SpeechRecognizerConfig) {
    super(config);
    this.backendType = config.backend ?? 'wasm';
  }

  async load(): Promise<void> {
    const backend = new TfjsBackend(this.backendType);
    await backend.ready();

    const configText = this.config.configSource ?? await resolveText(this.config.configPath);
    const modelConfig = parseModelConfig(configText);
    const modelSource = this.config.modelSource ?? this.config.modelPath;
    if (!modelSource) {
      throw new Error('Missing model source. Provide modelPath or modelSource.');
    }

    const vocabText = Array.isArray(this.config.vocabSource)
      ? JSON.stringify(this.config.vocabSource)
      : (this.config.vocabSource ?? await resolveText(this.config.vocabPath));

    const rawWeights = await loadSafeTensors(modelSource, backend);
    const summary = summarizeModelWeights(rawWeights);
    const mapped = mapWeights(rawWeights, modelConfig);

    const featurePipeline = new FeaturePipeline(modelConfig, backend);
    const encoder = new FastConformerEncoder(backend, mapped.encoder, modelConfig);
    const decoder = createDecoder(modelConfig, backend, mapped.decoder);
    const tokenizer = new SentencePieceDecoder(vocabText);
    const resampler = new Resampler(this.sampleRate, modelConfig.sampleRate);
    const cacheManager = new CacheManager();
    const endpointer = new Endpointer({ sampleRate: modelConfig.sampleRate });
    const chunkedInference = new ChunkedInference(
      modelConfig,
      backend,
      featurePipeline,
      encoder,
      decoder,
      cacheManager,
      endpointer,
    );

    this.backend = backend;
    this.modelConfig = modelConfig;
    this.featurePipeline = featurePipeline;
    this.encoder = encoder;
    this.decoder = decoder;
    this.tokenizer = tokenizer;
    this.resampler = resampler;
    this.chunkedInference = chunkedInference;
    this.loaded = true;

    this.emit('ready', {
      decoderType: modelConfig.decoderType,
      weightSummary: summary,
      consumedWeights: mapped.consumedKeys.length,
      unusedWeights: mapped.unusedKeys.length,
    });
  }

  processFrame(pcm: Float32Array): void {
    void this.processFrameInternal(pcm).catch((error: unknown) => {
      this.emit('error', error);
    });
  }

  async transcribe(audio: Float32Array): Promise<ASRResult> {
    this.ensureLoaded();
    const start = nowMs();

    const backend = this.backend as ComputeBackend;
    const resampled = this.resampler ? this.resampler.resample(audio) : audio;
    const mel = this.featurePipeline!.extractFeatures(resampled);
    const encoded = this.encoder!.forward(mel);
    const decoded = await this.decoder!.decode(encoded);
    const text = this.tokenizer!.decode(decoded.tokenIds);

    backend.dispose(mel);
    backend.dispose(encoded);
    this.decoder!.disposeState(decoded.state);

    return {
      text,
      isFinal: true,
      latencyMs: nowMs() - start,
      decoderType: this.modelConfig!.decoderType,
    };
  }

  override reset(): void {
    super.reset();
    this.tokenBuffer = [];
    this.chunkedInference?.reset();
    this.resampler?.reset();
  }

  dispose(): void {
    this.chunkedInference?.dispose();
    this.chunkedInference = null;
    this.backend = null;
    this.loaded = false;
  }

  private async processFrameInternal(pcm: Float32Array): Promise<void> {
    this.ensureLoaded();
    if (!this.chunkedInference || !this.resampler || !this.tokenizer || !this.modelConfig) {
      return;
    }

    const start = nowMs();
    const resampled = this.resampler.resampleStreaming(pcm);
    if (resampled.length === 0) {
      return;
    }

    const result = await this.chunkedInference.processFrame(resampled);
    if (result.tokenIds.length > 0) {
      this.tokenBuffer.push(...result.tokenIds);
      const text = this.tokenizer.decode(this.tokenBuffer);
      const partial: StreamingASRResult = {
        text,
        tokenIds: [...this.tokenBuffer],
        isFinal: false,
        latencyMs: nowMs() - start,
        decoderType: this.modelConfig.decoderType,
      };
      this.emit('partial', partial);
    }

    if (result.endpoint === 'speech-end' && this.tokenBuffer.length > 0) {
      const text = this.tokenizer.decode(this.tokenBuffer);
      const finalResult: StreamingASRResult = {
        text,
        tokenIds: [...this.tokenBuffer],
        isFinal: true,
        latencyMs: nowMs() - start,
        decoderType: this.modelConfig.decoderType,
      };
      this.emit('final', finalResult);
      this.tokenBuffer = [];
    }
  }

  private ensureLoaded(): void {
    if (!this.loaded) {
      throw new Error('SpeechRecognizer is not loaded. Call load() first.');
    }
  }
}
