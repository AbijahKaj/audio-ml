import { BaseApplication, type ApplicationConfig } from '../base/BaseApplication.js';
import { TfjsBackend } from '../../asr/compute/TfjsBackend.js';
import { parseModelConfig, type FastConformerConfig } from '../../asr/model/ModelConfig.js';
import { loadSafeTensors } from '../../asr/model/SafeTensorsLoader.js';
import { mapWeights } from '../../asr/model/WeightMapper.js';
import { FeaturePipeline } from '../../asr/features/FeaturePipeline.js';
import { Resampler } from '../../asr/features/Resampler.js';
import { FastConformerEncoder } from '../../asr/encoder/FastConformerEncoder.js';
import { createDecoder } from '../../asr/decoder/createDecoder.js';
import { SentencePieceDecoder } from '../../asr/text/SentencePieceDecoder.js';
import { ChunkedInference } from '../../asr/streaming/ChunkedInference.js';
import { Endpointer, type EndpointerConfig } from '../../asr/streaming/Endpointer.js';
import type { StreamingCache } from '../../asr/streaming/CacheManager.js';
import type { RNNTGreedyDecoder } from '../../asr/decoder/RNNTGreedyDecoder.js';
import type { TDTGreedyDecoder } from '../../asr/decoder/TDTGreedyDecoder.js';
import type { ComputeBackend } from '../../asr/compute/Backend.js';
import type { DecoderType } from '../../asr/model/ModelConfig.js';

export interface SpeechRecognizerConfig extends ApplicationConfig {
  /** URL or ArrayBuffer for the .safetensors model weights */
  modelPath: string | ArrayBuffer;
  /** URL or JSON string for the model_config.json */
  configPath: string;
  /** URL or JSON string for the vocab.json */
  vocabPath: string;
  /** Compute backend to use (default: 'cpu' for Node.js compatibility) */
  backend?: 'wasm' | 'webgpu' | 'webgl' | 'cpu';
  /** Streaming chunk size in samples (default: 16000 = 1 second at 16kHz) */
  chunkSizeSamples?: number;
}

export interface ASRResult {
  text: string;
  isFinal: boolean;
  latencyMs: number;
  decoderType: DecoderType;
}

export interface ASRPartialResult {
  text: string;
  isFinal: false;
}

export interface ASRFinalResult {
  text: string;
  isFinal: true;
  latencyMs: number;
  decoderType: DecoderType;
}

/**
 * SpeechRecognizer — main ASR application.
 *
 * Integrates the full pipeline:
 *   PCM audio → mel features → FastConformer encoder → RNNT/TDT decoder → text
 *
 * Supports both:
 *   - Offline mode: `await recognizer.transcribe(audioFloat32Array)`
 *   - Streaming mode: `recognizer.processFrame(chunk)` + events
 *
 * Events emitted:
 *   'ready'    — model loaded, { decoderType }
 *   'partial'  — streaming partial result, { text }
 *   'final'    — streaming final result after speech-end, { text, latencyMs, decoderType }
 *   'error'    — error during load or transcription, { error }
 *
 * Usage:
 * ```typescript
 * import { SpeechRecognizer } from 'audio-ml/applications';
 *
 * const recognizer = new SpeechRecognizer({
 *   sampleRate: 16000,
 *   modelPath: './parakeet_120m.safetensors',
 *   configPath: './model_config.json',
 *   vocabPath: './vocab.json',
 *   backend: 'wasm',
 * });
 *
 * await recognizer.load();
 * const result = await recognizer.transcribe(audioFloat32Array);
 * console.log(result.text);
 * ```
 */
export class SpeechRecognizer extends BaseApplication {
  private backend!: ComputeBackend;
  private modelConfig!: FastConformerConfig;
  private featurePipeline!: FeaturePipeline;
  private encoder!: FastConformerEncoder;
  private decoder!: RNNTGreedyDecoder | TDTGreedyDecoder;
  private tokenizer!: SentencePieceDecoder;
  private chunkedInference!: ChunkedInference;
  private endpointer!: Endpointer;
  private resampler!: Resampler;
  private isLoaded = false;

  private streamingCache: StreamingCache | null = null;
  private utteranceStartTime = 0;
  private readonly backendType: 'wasm' | 'webgpu' | 'webgl' | 'cpu';

  private readonly modelPath: string | ArrayBuffer;
  private readonly configPath: string;
  private readonly vocabPath: string;

  constructor(config: SpeechRecognizerConfig) {
    super(config);
    this.modelPath = config.modelPath;
    this.configPath = config.configPath;
    this.vocabPath = config.vocabPath;
    this.backendType = config.backend ?? 'cpu';
  }

  /**
   * Load model weights and initialize all components.
   * Must be called before transcribe() or processFrame().
   */
  async load(): Promise<void> {
    try {
      this.backend = new TfjsBackend(this.backendType);

      // Load config
      const configText = await fetchText(this.configPath);
      this.modelConfig = parseModelConfig(configText);

      // Load weights
      const weights = await loadSafeTensors(this.modelPath, this.backend);
      const modelWeights = mapWeights(weights, this.modelConfig);

      // Initialize components
      this.featurePipeline = new FeaturePipeline(this.modelConfig);
      this.resampler = new Resampler(this.sampleRate, this.modelConfig.sampleRate);
      this.encoder = new FastConformerEncoder(this.backend, modelWeights.encoder, this.modelConfig);
      this.decoder = createDecoder(this.modelConfig, this.backend, modelWeights.decoder);

      const vocabText = await fetchText(this.vocabPath);
      this.tokenizer = new SentencePieceDecoder(vocabText);

      this.chunkedInference = new ChunkedInference(
        this.backend,
        this.encoder,
        this.decoder,
        this.featurePipeline,
        this.tokenizer,
        this.modelConfig,
      );

      this.endpointer = new Endpointer({
        sampleRate: this.modelConfig.sampleRate,
        endpointSilenceDurationMs: 800,
        minSpeechDurationMs: 300,
      } as EndpointerConfig);

      this.isLoaded = true;
      this.emit('ready', { decoderType: this.modelConfig.decoderType });
    } catch (error) {
      this.emit('error', { error });
      throw error;
    }
  }

  /**
   * Offline transcription — process a complete audio buffer.
   *
   * @param audio  Float32Array of PCM samples at `sampleRate` Hz
   * @returns      Transcription result with text, latency, and decoder type
   */
  async transcribe(audio: Float32Array): Promise<ASRResult> {
    if (!this.isLoaded) throw new Error('Model not loaded. Call load() first.');

    const start = performance.now();

    // Resample if needed
    const audio16k = this.resampler.resample(audio);

    // Extract features
    const { features, numFrames } = this.featurePipeline.extractFeatures(audio16k);

    // Encode
    const encoded = this.encoder.forward(features, numFrames);

    // Decode
    const tokenIds = await this.decoder.decode(encoded);
    this.backend.dispose(encoded);

    const text = this.tokenizer.decode(tokenIds);
    const latencyMs = performance.now() - start;

    return {
      text,
      isFinal: true,
      latencyMs,
      decoderType: this.modelConfig.decoderType,
    };
  }

  /**
   * Streaming frame processing.
   * Call this with each audio chunk from a microphone stream.
   *
   * Emits 'partial' events during speech and 'final' when an utterance ends.
   *
   * @param pcm  Float32Array of PCM samples (at the configured sampleRate)
   */
  processFrame(pcm: Float32Array): void {
    if (!this.isLoaded) return;

    const pcm16k = this.resampler.resample(pcm);
    const event = this.endpointer.processFrame(pcm16k);

    if (event === 'speech' || event === 'silence') {
      if (event === 'speech' && this.streamingCache === null) {
        this.streamingCache = this.chunkedInference.startUtterance();
        this.utteranceStartTime = performance.now();
      }

      if (this.streamingCache !== null) {
        // Process asynchronously — don't block the audio thread
        this.processChunkAsync(pcm16k, this.streamingCache);
      }
    } else if (event === 'speech-end' && this.streamingCache !== null) {
      const cache = this.streamingCache;
      this.streamingCache = null;
      this.finalizeAsync(cache);
    }
  }

  private async processChunkAsync(
    chunk: Float32Array,
    cache: StreamingCache,
  ): Promise<void> {
    try {
      const result = await this.chunkedInference.processChunk(chunk, cache);
      if (result.newTokens.length > 0) {
        this.emit('partial', { text: result.partialText, isFinal: false } satisfies ASRPartialResult);
      }
    } catch (error) {
      this.emit('error', { error });
    }
  }

  private async finalizeAsync(cache: StreamingCache): Promise<void> {
    try {
      const { text } = this.chunkedInference.finalizeUtterance(cache);
      const latencyMs = performance.now() - this.utteranceStartTime;
      this.chunkedInference.disposeCache(cache);
      this.emit('final', {
        text,
        isFinal: true,
        latencyMs,
        decoderType: this.modelConfig.decoderType,
      } satisfies ASRFinalResult);
    } catch (error) {
      this.emit('error', { error });
    }
  }

  override reset(): void {
    super.reset();
    if (this.streamingCache) {
      this.chunkedInference.disposeCache(this.streamingCache);
      this.streamingCache = null;
    }
    if (this.endpointer) {
      this.endpointer.reset();
    }
  }

  get loaded(): boolean {
    return this.isLoaded;
  }

  get decoderType(): DecoderType | null {
    return this.isLoaded ? this.modelConfig.decoderType : null;
  }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

async function fetchText(source: string): Promise<string> {
  // Try fetch first (browser + Node 18+)
  if (typeof fetch !== 'undefined') {
    return (await fetch(source)).text();
  }
  // Fallback to Node.js fs
  const { readFile } = await import('fs/promises');
  return readFile(source, 'utf8');
}
