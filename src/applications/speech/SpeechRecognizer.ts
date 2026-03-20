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
import { NemoPythonBridge } from '../../asr/bridge/NemoPythonBridge';

type BackendType = 'wasm' | 'webgpu' | 'webgl' | 'cpu';
type ModelSource = string | ArrayBuffer | Uint8Array;
type VocabSource = string | string[];
type InferenceEngine = 'tfjs' | 'nemo-python';

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
  engine?: InferenceEngine;
  nemoModelName?: string;
  nemoPythonPath?: string;
  nemoBridgeScriptPath?: string;
  streamingPartialIntervalMs?: number;
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
  private readonly engine: InferenceEngine;
  private backend: ComputeBackend | null = null;
  private modelConfig: FastConformerConfig | null = null;
  private featurePipeline: FeaturePipeline | null = null;
  private encoder: FastConformerEncoder | null = null;
  private decoder: TransducerDecoder | null = null;
  private tokenizer: SentencePieceDecoder | null = null;
  private resampler: Resampler | null = null;
  private chunkedInference: ChunkedInference | null = null;
  private endpointer: Endpointer | null = null;
  private nemoBridge: NemoPythonBridge | null = null;
  private nemoModelSampleRate = 16000;
  private decoderType: DecoderType = 'rnnt';
  private streamingPartialIntervalSamples = 6400;
  private bufferedStreamingAudio: Float32Array[] = [];
  private bufferedStreamingSamples = 0;
  private lastPartialText = '';
  private processingQueue: Promise<void> = Promise.resolve();
  private tokenBuffer: number[] = [];
  private loaded = false;

  constructor(private config: SpeechRecognizerConfig) {
    super(config);
    this.backendType = config.backend ?? 'wasm';
    this.engine = config.engine ?? 'tfjs';
  }

  async load(): Promise<void> {
    if (this.engine === 'nemo-python') {
      await this.loadNemoBridgeEngine();
      return;
    }

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
    this.endpointer = endpointer;
    this.loaded = true;
    this.decoderType = modelConfig.decoderType;

    this.emit('ready', {
      decoderType: modelConfig.decoderType,
      weightSummary: summary,
      consumedWeights: mapped.consumedKeys.length,
      unusedWeights: mapped.unusedKeys.length,
    });
  }

  processFrame(pcm: Float32Array): void {
    if (this.engine === 'nemo-python') {
      this.processingQueue = this.processingQueue
        .then(() => this.processFrameWithNemo(pcm))
        .catch((error: unknown) => {
          this.emit('error', error);
        });
      return;
    }

    void this.processFrameInternal(pcm).catch((error: unknown) => {
      this.emit('error', error);
    });
  }

  async transcribe(audio: Float32Array): Promise<ASRResult> {
    this.ensureLoaded();
    const start = nowMs();

    if (this.engine === 'nemo-python') {
      const resampled = this.resampler ? this.resampler.resample(audio) : audio;
      const text = await this.nemoBridge!.transcribePcm(resampled, this.nemoModelSampleRate);
      return {
        text,
        isFinal: true,
        latencyMs: nowMs() - start,
        decoderType: this.decoderType,
      };
    }

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
      decoderType: this.decoderType,
    };
  }

  override reset(): void {
    super.reset();
    this.tokenBuffer = [];
    this.lastPartialText = '';
    this.bufferedStreamingAudio = [];
    this.bufferedStreamingSamples = 0;
    this.chunkedInference?.reset();
    this.endpointer?.reset();
    this.resampler?.reset();
  }

  dispose(): void {
    this.chunkedInference?.dispose();
    this.chunkedInference = null;
    if (this.nemoBridge) {
      void this.nemoBridge.stop();
      this.nemoBridge = null;
    }
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
        decoderType: this.decoderType,
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
        decoderType: this.decoderType,
      };
      this.emit('final', finalResult);
      this.tokenBuffer = [];
    }
  }

  private async processFrameWithNemo(pcm: Float32Array): Promise<void> {
    this.ensureLoaded();
    if (!this.nemoBridge || !this.resampler || !this.endpointer) {
      return;
    }

    const start = nowMs();
    const resampled = this.resampler.resampleStreaming(pcm);
    if (resampled.length === 0) {
      return;
    }

    this.bufferedStreamingAudio.push(resampled);
    this.bufferedStreamingSamples += resampled.length;
    const endpoint = this.endpointer.processFrame(resampled);
    const shouldDecode = this.bufferedStreamingSamples >= this.streamingPartialIntervalSamples
      || endpoint === 'speech-end';
    if (!shouldDecode) {
      return;
    }

    const audio = this.concatFloat32Arrays(this.bufferedStreamingAudio);
    const text = await this.nemoBridge.transcribePcm(audio, this.nemoModelSampleRate);

    if (endpoint === 'speech-end') {
      const finalResult: StreamingASRResult = {
        text,
        tokenIds: [],
        isFinal: true,
        latencyMs: nowMs() - start,
        decoderType: this.decoderType,
      };
      this.emit('final', finalResult);
      this.bufferedStreamingAudio = [];
      this.bufferedStreamingSamples = 0;
      this.lastPartialText = '';
      this.endpointer.reset();
      this.resampler.reset();
      return;
    }

    if (text !== this.lastPartialText) {
      this.lastPartialText = text;
      const partialResult: StreamingASRResult = {
        text,
        tokenIds: [],
        isFinal: false,
        latencyMs: nowMs() - start,
        decoderType: this.decoderType,
      };
      this.emit('partial', partialResult);
    }
  }

  private async loadNemoBridgeEngine(): Promise<void> {
    const modelName = this.config.nemoModelName ?? this.config.modelPath;
    if (!modelName) {
      throw new Error('nemo-python engine requires nemoModelName (or modelPath as model name).');
    }

    const bridge = new NemoPythonBridge({
      pythonPath: this.config.nemoPythonPath,
      scriptPath: this.config.nemoBridgeScriptPath,
    });
    const ready = await bridge.start(modelName);
    this.nemoBridge = bridge;
    this.decoderType = ready.decoderType;
    this.nemoModelSampleRate = ready.sampleRate;
    this.resampler = new Resampler(this.sampleRate, ready.sampleRate);
    this.endpointer = new Endpointer({ sampleRate: ready.sampleRate });
    const intervalMs = this.config.streamingPartialIntervalMs ?? 400;
    this.streamingPartialIntervalSamples = Math.max(
      Math.round((ready.sampleRate * intervalMs) / 1000),
      Math.round(ready.sampleRate * 0.1),
    );
    this.loaded = true;

    this.emit('ready', {
      decoderType: this.decoderType,
      engine: 'nemo-python',
      modelName: ready.modelName,
      sampleRate: ready.sampleRate,
    });
  }

  private ensureLoaded(): void {
    if (!this.loaded) {
      throw new Error('SpeechRecognizer is not loaded. Call load() first.');
    }
  }

  private concatFloat32Arrays(chunks: Float32Array[]): Float32Array {
    if (chunks.length === 0) {
      return new Float32Array(0);
    }
    if (chunks.length === 1) {
      return chunks[0];
    }
    let length = 0;
    for (const chunk of chunks) {
      length += chunk.length;
    }
    const merged = new Float32Array(length);
    let offset = 0;
    for (const chunk of chunks) {
      merged.set(chunk, offset);
      offset += chunk.length;
    }
    return merged;
  }
}
