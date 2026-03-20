import { BaseApplication, type ApplicationConfig } from '../base/BaseApplication.js';
import type { VADConfig } from './VAD.js';

import { TfjsBackend } from '../../asr/compute/TfjsBackend.js';
import type { BackendKind } from '../../asr/compute/types.js';
import { createDecoder } from '../../asr/decoder/createDecoder.js';
import { FastConformerEncoder } from '../../asr/encoder/FastConformerEncoder.js';
import { FeaturePipeline } from '../../asr/features/FeaturePipeline.js';
import { Resampler } from '../../asr/features/Resampler.js';
import { parseModelConfig, type DecoderType, type FastConformerConfig } from '../../asr/model/ModelConfig.js';
import { loadSafeTensors } from '../../asr/model/SafeTensorsLoader.js';
import { mapWeights } from '../../asr/model/WeightMapper.js';
import { CacheManager, type StreamingCache } from '../../asr/streaming/CacheManager.js';
import { ChunkedInference } from '../../asr/streaming/ChunkedInference.js';
import { Endpointer } from '../../asr/streaming/Endpointer.js';
import { SentencePieceDecoder } from '../../asr/text/SentencePieceDecoder.js';
import type { RNNTGreedyDecoder } from '../../asr/decoder/RNNTGreedyDecoder.js';
import type { TDTGreedyDecoder } from '../../asr/decoder/TDTGreedyDecoder.js';

export interface ASRResult {
  text: string;
  isFinal: boolean;
  latencyMs: number;
  decoderType: DecoderType | 'transformer';
  tokens: number[];
}

export type SpeechRecognizerProvider = 'transformers-js' | 'fastconformer-tfjs';
export type TransformersDtype = 'auto' | 'fp32' | 'fp16' | 'q8' | 'int8' | 'uint8' | 'q4' | 'bnb4' | 'q4f16';
type TransformersPipelineHandle = (audio: Float32Array, options?: Record<string, unknown>) => Promise<{ text?: string }>;

export interface SpeechRecognizerConfig extends ApplicationConfig {
  provider?: SpeechRecognizerProvider;
  modelPath?: string;
  configPath?: string;
  vocabPath?: string;
  modelId?: string;
  backend?: BackendKind;
  chunkDurationMs?: number;
  vadConfig?: Omit<VADConfig, 'sampleRate'>;
  language?: string;
  whisperTask?: 'transcribe' | 'translate';
  chunkLengthSeconds?: number;
  strideLengthSeconds?: number;
  partialDecodeEveryChunks?: number;
  transformersDtype?: TransformersDtype | Record<string, TransformersDtype>;
}

function concatFloat32(chunks: Float32Array[]): Float32Array {
  const total = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const output = new Float32Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    output.set(chunk, offset);
    offset += chunk.length;
  }
  return output;
}

async function fetchText(path: string): Promise<string> {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`Failed to fetch ${path}: ${response.status} ${response.statusText}`);
  }
  return response.text();
}

export class SpeechRecognizer extends BaseApplication {
  private provider: SpeechRecognizerProvider;
  private backendType: BackendKind;
  private backendInstance: TfjsBackend | null = null;
  private modelConfig: FastConformerConfig | null = null;
  private featurePipeline: FeaturePipeline | null = null;
  private encoder: FastConformerEncoder | null = null;
  private decoder: RNNTGreedyDecoder | TDTGreedyDecoder | null = null;
  private tokenizer: SentencePieceDecoder | null = null;
  private resampler: Resampler | null = null;
  private endpointer: Endpointer | null = null;
  private chunkedInference: ChunkedInference | null = null;
  private cacheManager = new CacheManager();
  private streamingCache: StreamingCache | null = null;
  private transformersPipeline: TransformersPipelineHandle | null = null;
  private transformersModelId: string | null = null;
  private utteranceAudio: Float32Array[] = [];
  private utteranceActive = false;
  private partialChunkCounter = 0;
  private lastPartialText = '';
  private decodeChain: Promise<void> = Promise.resolve();
  private isLoaded = false;

  constructor(private readonly config: SpeechRecognizerConfig) {
    super(config);
    this.provider = config.provider ?? (config.modelPath && config.configPath && config.vocabPath ? 'fastconformer-tfjs' : 'transformers-js');
    this.backendType = config.backend ?? 'wasm';
  }

  async load(): Promise<void> {
    if (this.provider === 'transformers-js') {
      await this.loadTransformersPipeline();
      return;
    }

    await this.loadFastConformerPipeline();
  }

  processFrame(pcm: Float32Array): { isLoaded: boolean; active: boolean } {
    if (!this.isLoaded || !this.resampler || !this.endpointer || !this.chunkedInference) {
      return { isLoaded: false, active: false };
    }

    const endpointState = this.endpointer.processFrame(pcm);
    const resampled = this.resampler.resample(pcm);

    if (endpointState === 'speech' || endpointState === 'speech-end') {
      if (!this.utteranceActive) {
        this.utteranceActive = true;
        this.utteranceAudio = [];
        this.chunkedInference.reset();
        this.partialChunkCounter = 0;
      }

      this.utteranceAudio.push(resampled);
      const chunks = this.chunkedInference.push(resampled);
      if (chunks.length > 0) {
        this.partialChunkCounter += chunks.length;
        if (this.partialChunkCounter >= (this.config.partialDecodeEveryChunks ?? 4)) {
          this.partialChunkCounter = 0;
          this.queueDecode(false);
        }
      }
    }

    if (endpointState === 'speech-end' && this.utteranceActive) {
      this.chunkedInference.flush();
      this.queueDecode(true);
    }

    return { isLoaded: true, active: this.utteranceActive };
  }

  async transcribe(audio: Float32Array): Promise<ASRResult> {
    if (!this.isLoaded) {
      throw new Error('SpeechRecognizer is not loaded');
    }

    if (this.provider === 'transformers-js') {
      return this.transcribeWithTransformers(audio);
    }

    return this.transcribeWithFastConformer(audio);
  }

  async finalize(): Promise<ASRResult | null> {
    if (!this.utteranceActive) {
      return null;
    }

    const audio = concatFloat32(this.utteranceAudio);
    if (audio.length === 0) {
      this.reset();
      return null;
    }

    const result = await this.transcribe(audio);
    this.emit('final', { ...result, isFinal: true });
    this.reset();
    return result;
  }

  override reset(): void {
    super.reset();
    if (this.streamingCache && this.backendInstance) {
      this.cacheManager.dispose(this.streamingCache, this.backendInstance);
    }
    this.streamingCache = this.modelConfig ? this.cacheManager.create(this.modelConfig) : null;
    this.featurePipeline?.reset();
    this.chunkedInference?.reset();
    this.endpointer?.reset();
    this.utteranceAudio = [];
    this.utteranceActive = false;
    this.partialChunkCounter = 0;
    this.lastPartialText = '';
  }

  private async loadFastConformerPipeline(): Promise<void> {
    if (!this.config.modelPath || !this.config.configPath || !this.config.vocabPath) {
      throw new Error('FastConformer loading requires modelPath, configPath, and vocabPath');
    }

    this.backendInstance = new TfjsBackend(this.backendType);
    await this.backendInstance.ready();

    const parsedConfig = parseModelConfig(await fetchText(this.config.configPath));
    const safeTensors = await loadSafeTensors(this.config.modelPath, this.backendInstance);
    const modelWeights = mapWeights(safeTensors, parsedConfig);

    this.modelConfig = parsedConfig;
    this.featurePipeline = new FeaturePipeline(parsedConfig, this.backendInstance);
    this.encoder = new FastConformerEncoder(this.backendInstance, modelWeights.encoder, parsedConfig);
    this.decoder = createDecoder(parsedConfig, this.backendInstance, modelWeights.decoder);
    this.tokenizer = new SentencePieceDecoder(await fetchText(this.config.vocabPath));
    this.resampler = new Resampler(this.config.sampleRate, parsedConfig.sampleRate);
    this.endpointer = new Endpointer({
      sampleRate: this.config.sampleRate,
      ...(this.config.vadConfig ?? {}),
    });
    this.chunkedInference = new ChunkedInference({
      sampleRate: parsedConfig.sampleRate,
      chunkDurationMs: this.config.chunkDurationMs,
    });
    this.streamingCache = this.cacheManager.create(parsedConfig);
    this.partialChunkCounter = 0;
    this.isLoaded = true;
    this.emit('ready', {
      provider: this.provider,
      decoderType: parsedConfig.decoderType,
      unconsumedWeights: modelWeights.unconsumed,
    });
  }

  private async loadTransformersPipeline(): Promise<void> {
    const { env, pipeline } = await import('@huggingface/transformers');
    const isBrowser = typeof globalThis !== 'undefined' && 'window' in globalThis;
    env.allowLocalModels = !isBrowser;

    if (isBrowser) {
      env.useBrowserCache = true;
    }

    this.resampler = new Resampler(this.config.sampleRate, 16000);
    this.endpointer = new Endpointer({
      sampleRate: this.config.sampleRate,
      ...(this.config.vadConfig ?? {}),
    });
    this.chunkedInference = new ChunkedInference({
      sampleRate: 16000,
      chunkDurationMs: this.config.chunkDurationMs,
    });

    const modelId = this.config.modelId ?? 'Xenova/wav2vec2-base-960h';
    this.transformersModelId = modelId;
    this.transformersPipeline = await pipeline(
      'automatic-speech-recognition',
      modelId,
      {
        dtype: this.config.transformersDtype,
        progress_callback: (info: unknown) => this.emit('progress', info),
      },
    ) as unknown as TransformersPipelineHandle;

    this.partialChunkCounter = 0;
    this.isLoaded = true;
    this.emit('ready', {
      provider: this.provider,
      decoderType: 'transformer',
      modelId,
    });
  }

  private async transcribeWithFastConformer(audio: Float32Array): Promise<ASRResult> {
    if (!this.isLoaded || !this.backendInstance || !this.featurePipeline || !this.encoder || !this.decoder || !this.tokenizer || !this.modelConfig) {
      throw new Error('SpeechRecognizer is not loaded');
    }

    const startedAt = typeof performance !== 'undefined' ? performance.now() : Date.now();
    const mel = this.featurePipeline.extractFeatures(audio);
    const encoded = this.encoder.forward(mel);
    const tokenIds = await this.decoder.decode(encoded);
    const text = this.tokenizer.decode(tokenIds);
    this.backendInstance.dispose(mel);
    this.backendInstance.dispose(encoded);
    const finishedAt = typeof performance !== 'undefined' ? performance.now() : Date.now();

    return {
      text,
      isFinal: true,
      latencyMs: finishedAt - startedAt,
      decoderType: this.modelConfig.decoderType,
      tokens: tokenIds,
    };
  }

  private async transcribeWithTransformers(audio: Float32Array): Promise<ASRResult> {
    if (!this.transformersPipeline || !this.resampler) {
      throw new Error('Transformers.js pipeline is not loaded');
    }

    const startedAt = typeof performance !== 'undefined' ? performance.now() : Date.now();
    const resampled = this.resampler.resample(audio);
    const options: Record<string, unknown> = {
      chunk_length_s: this.config.chunkLengthSeconds ?? 10,
      stride_length_s: this.config.strideLengthSeconds ?? 2,
    };
    const isEnglishOnlyModel = this.transformersModelId?.endsWith('.en') ?? false;

    if (this.config.language && !isEnglishOnlyModel) {
      options.language = this.config.language;
    }

    if (this.config.whisperTask && !isEnglishOnlyModel) {
      options.task = this.config.whisperTask;
    }

    const result = await this.transformersPipeline(resampled, options);
    const finishedAt = typeof performance !== 'undefined' ? performance.now() : Date.now();

    return {
      text: result.text ?? '',
      isFinal: true,
      latencyMs: finishedAt - startedAt,
      decoderType: 'transformer',
      tokens: [],
    };
  }

  private queueDecode(isFinal: boolean): void {
    this.decodeChain = this.decodeChain
      .then(async () => {
        const audio = concatFloat32(this.utteranceAudio);
        if (audio.length === 0) {
          return;
        }

        const result = await this.transcribe(audio);
        const eventName = isFinal ? 'final' : 'partial';
        const payload = { ...result, isFinal };

        if (isFinal || result.text !== this.lastPartialText) {
          this.emit(eventName, payload);
        }

        this.lastPartialText = result.text;

        if (isFinal) this.reset();
      })
      .catch((error: unknown) => {
        this.emit('error', error);
      });
  }
}
