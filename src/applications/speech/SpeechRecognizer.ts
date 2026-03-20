import { BaseApplication, type ApplicationConfig } from '../base/BaseApplication';
import type { VADConfig } from './VAD';
import { TfjsBackend } from '../../asr/compute/TfjsBackend';
import type { BackendKind } from '../../asr/compute/types';
import { createDecoder } from '../../asr/decoder/createDecoder';
import { FastConformerEncoder } from '../../asr/encoder/FastConformerEncoder';
import { FeaturePipeline } from '../../asr/features/FeaturePipeline';
import { Resampler } from '../../asr/features/Resampler';
import { parseModelConfig, type DecoderType, type FastConformerConfig } from '../../asr/model/ModelConfig';
import { loadSafeTensors } from '../../asr/model/SafeTensorsLoader';
import { mapWeights } from '../../asr/model/WeightMapper';
import { CacheManager, type StreamingCache } from '../../asr/streaming/CacheManager';
import { ChunkedInference } from '../../asr/streaming/ChunkedInference';
import { Endpointer } from '../../asr/streaming/Endpointer';
import { SentencePieceDecoder } from '../../asr/text/SentencePieceDecoder';
import type { RNNTGreedyDecoder } from '../../asr/decoder/RNNTGreedyDecoder';
import type { TDTGreedyDecoder } from '../../asr/decoder/TDTGreedyDecoder';

export interface ASRResult {
  text: string;
  isFinal: boolean;
  latencyMs: number;
  decoderType: DecoderType;
  tokens: number[];
}

export interface SpeechRecognizerConfig extends ApplicationConfig {
  modelPath: string;
  configPath: string;
  vocabPath: string;
  backend?: BackendKind;
  chunkDurationMs?: number;
  vadConfig?: Omit<VADConfig, 'sampleRate'>;
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
  private utteranceAudio: Float32Array[] = [];
  private utteranceActive = false;
  private lastPartialText = '';
  private decodeChain: Promise<void> = Promise.resolve();
  private isLoaded = false;

  constructor(private readonly config: SpeechRecognizerConfig) {
    super(config);
    this.backendType = config.backend ?? 'wasm';
  }

  async load(): Promise<void> {
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
    this.isLoaded = true;
    this.emit('ready', { decoderType: parsedConfig.decoderType, unconsumedWeights: modelWeights.unconsumed });
  }

  processFrame(pcm: Float32Array): { isLoaded: boolean; active: boolean } {
    if (!this.isLoaded || !this.modelConfig || !this.resampler || !this.endpointer || !this.chunkedInference) {
      return { isLoaded: false, active: false };
    }

    const endpointState = this.endpointer.processFrame(pcm);
    const resampled = this.resampler.resample(pcm);

    if (endpointState === 'speech' || endpointState === 'speech-end') {
      if (!this.utteranceActive) {
        this.utteranceActive = true;
        this.utteranceAudio = [];
        this.chunkedInference.reset();
      }

      this.utteranceAudio.push(resampled);
      const chunks = this.chunkedInference.push(resampled);
      if (chunks.length > 0) {
        this.queueDecode(false);
      }
    }

    if (endpointState === 'speech-end' && this.utteranceActive) {
      this.chunkedInference.flush();
      this.queueDecode(true);
    }

    return { isLoaded: true, active: this.utteranceActive };
  }

  async transcribe(audio: Float32Array): Promise<ASRResult> {
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
    this.lastPartialText = '';
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

        if (isFinal) {
          this.reset();
        }
      })
      .catch((error: unknown) => {
        this.emit('error', error);
      });
  }
}
