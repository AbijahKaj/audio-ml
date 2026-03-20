import type { ComputeBackend } from '../compute/Backend';
import type { FastConformerConfig } from '../model/ModelConfig';
import { FeaturePipeline } from '../features/FeaturePipeline';
import { FastConformerEncoder } from '../encoder/FastConformerEncoder';
import type { RNNTGreedyDecoder } from '../decoder/RNNTGreedyDecoder';
import type { TDTGreedyDecoder } from '../decoder/TDTGreedyDecoder';
import { SentencePieceDecoder } from '../text/SentencePieceDecoder';
import { Resampler } from '../features/Resampler';

export interface StreamingResult {
  text: string;
  isFinal: boolean;
  isPartial: boolean;
  latencyMs: number;
  decoderType: 'rnnt' | 'tdt';
}

export interface ChunkedInferenceConfig {
  chunkSizeMs: number;
  maxContextFrames: number;
  inputSampleRate?: number;
}

interface StreamingState {
  audioBuffer: Float32Array;
  processedAudio: Float32Array | null;
  allTokens: number[];
  totalAudioProcessed: number;
}

/**
 * Chunked streaming inference engine.
 * Buffers incoming PCM, extracts features in chunks, runs encoder + decoder incrementally.
 */
export class ChunkedInference {
  private backend: ComputeBackend;
  private config: FastConformerConfig;
  private featurePipeline: FeaturePipeline;
  private encoder: FastConformerEncoder;
  private decoder: RNNTGreedyDecoder | TDTGreedyDecoder;
  private tokenizer: SentencePieceDecoder;
  private resampler: Resampler | null;

  private chunkSizeSamples: number;
  private state: StreamingState;

  constructor(
    backend: ComputeBackend,
    config: FastConformerConfig,
    featurePipeline: FeaturePipeline,
    encoder: FastConformerEncoder,
    decoder: RNNTGreedyDecoder | TDTGreedyDecoder,
    tokenizer: SentencePieceDecoder,
    inferenceConfig: ChunkedInferenceConfig,
  ) {
    this.backend = backend;
    this.config = config;
    this.featurePipeline = featurePipeline;
    this.encoder = encoder;
    this.decoder = decoder;
    this.tokenizer = tokenizer;

    const inputSampleRate = inferenceConfig.inputSampleRate ?? config.sampleRate;
    this.chunkSizeSamples = Math.round(inputSampleRate * inferenceConfig.chunkSizeMs / 1000);

    if (inputSampleRate !== config.sampleRate) {
      this.resampler = new Resampler(inputSampleRate, config.sampleRate);
    } else {
      this.resampler = null;
    }

    this.state = this.createInitialState();
  }

  /**
   * Feed PCM audio data. Returns partial results when enough audio has accumulated.
   */
  async feedAudio(pcm: Float32Array): Promise<StreamingResult | null> {
    // Append to buffer
    const newBuffer = new Float32Array(this.state.audioBuffer.length + pcm.length);
    newBuffer.set(this.state.audioBuffer);
    newBuffer.set(pcm, this.state.audioBuffer.length);
    this.state.audioBuffer = newBuffer;

    // Process chunks when we have enough audio
    if (this.state.audioBuffer.length < this.chunkSizeSamples) {
      return null;
    }

    return this.processChunk();
  }

  /**
   * Force processing of remaining audio and return final result.
   */
  async flush(): Promise<StreamingResult> {
    if (this.state.audioBuffer.length > 0) {
      const result = await this.forceProcessRemaining();
      if (result) {
        return { ...result, isFinal: true };
      }
    }

    return {
      text: this.tokenizer.decode(this.state.allTokens),
      isFinal: true,
      isPartial: false,
      latencyMs: 0,
      decoderType: this.config.decoderType,
    };
  }

  /**
   * Reset all streaming state for a new utterance.
   */
  reset(): void {
    this.state = this.createInitialState();
  }

  get currentText(): string {
    return this.tokenizer.decode(this.state.allTokens);
  }

  get tokenCount(): number {
    return this.state.allTokens.length;
  }

  private async processChunk(): Promise<StreamingResult | null> {
    const start = performance.now();

    // Extract one chunk worth of audio but keep the full accumulated audio
    const chunkAudio = new Float32Array(this.chunkSizeSamples);
    chunkAudio.set(this.state.audioBuffer.subarray(0, this.chunkSizeSamples));
    const remaining = new Float32Array(this.state.audioBuffer.length - this.chunkSizeSamples);
    remaining.set(this.state.audioBuffer.subarray(this.chunkSizeSamples));
    this.state.audioBuffer = remaining;

    // Accumulate all processed audio for re-encoding
    const newProcessed = new Float32Array(this.state.totalAudioProcessed + chunkAudio.length);
    if (this.state.processedAudio) {
      newProcessed.set(this.state.processedAudio);
    }
    newProcessed.set(chunkAudio, this.state.totalAudioProcessed);
    this.state.processedAudio = newProcessed;
    this.state.totalAudioProcessed += chunkAudio.length;

    let audio: Float32Array = this.state.processedAudio;
    if (this.resampler) {
      audio = this.resampler.resample(audio);
    }

    // Run full offline pipeline on all accumulated audio
    const mel = this.featurePipeline.extractFeatures(audio);
    const encoded = this.encoder.forward(mel);
    const tokenIds = await this.decoder.decode(encoded);
    this.state.allTokens = tokenIds;

    this.backend.dispose(mel);
    this.backend.dispose(encoded);

    const latencyMs = performance.now() - start;

    return {
      text: this.tokenizer.decode(this.state.allTokens),
      isFinal: false,
      isPartial: true,
      latencyMs,
      decoderType: this.config.decoderType,
    };
  }

  private async forceProcessRemaining(): Promise<StreamingResult | null> {
    const start = performance.now();

    // Add remaining buffer to processed audio
    if (this.state.audioBuffer.length > 0) {
      const newProcessed = new Float32Array(this.state.totalAudioProcessed + this.state.audioBuffer.length);
      if (this.state.processedAudio) {
        newProcessed.set(this.state.processedAudio);
      }
      newProcessed.set(this.state.audioBuffer, this.state.totalAudioProcessed);
      this.state.processedAudio = newProcessed;
      this.state.totalAudioProcessed += this.state.audioBuffer.length;
      this.state.audioBuffer = new Float32Array(0);
    }

    if (!this.state.processedAudio || this.state.totalAudioProcessed === 0) {
      return null;
    }

    let audio: Float32Array = this.state.processedAudio;
    if (this.resampler) {
      audio = this.resampler.resample(audio);
    }

    const minSamples = this.featurePipeline.frameLength;
    if (audio.length < minSamples) {
      const padded = new Float32Array(minSamples);
      padded.set(audio);
      audio = padded;
    }

    const mel = this.featurePipeline.extractFeatures(audio);
    const encoded = this.encoder.forward(mel);
    const tokenIds = await this.decoder.decode(encoded);
    this.state.allTokens = tokenIds;

    this.backend.dispose(mel);
    this.backend.dispose(encoded);

    return {
      text: this.tokenizer.decode(this.state.allTokens),
      isFinal: true,
      isPartial: false,
      latencyMs: performance.now() - start,
      decoderType: this.config.decoderType,
    };
  }

  private createInitialState(): StreamingState {
    return {
      audioBuffer: new Float32Array(0),
      processedAudio: null,
      allTokens: [],
      totalAudioProcessed: 0,
    };
  }
}
