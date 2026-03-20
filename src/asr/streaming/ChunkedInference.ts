import type { ComputeBackend } from '../compute/Backend';
import type { TensorHandle } from '../compute/types';
import type { FastConformerConfig } from '../model/ModelConfig';
import { FeaturePipeline } from '../features/FeaturePipeline';
import { FastConformerEncoder, type StreamingEncoderState } from '../encoder/FastConformerEncoder';
import { RNNTGreedyDecoder } from '../decoder/RNNTGreedyDecoder';
import { TDTGreedyDecoder } from '../decoder/TDTGreedyDecoder';
import { SentencePieceDecoder } from '../text/SentencePieceDecoder';
import type { PredictionState } from '../decoder/PredictionNetwork';
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
  encoderState: StreamingEncoderState | null;
  decoderState: PredictionState | null;
  lastToken: number;
  tdtFrameOffset: number;
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
  private maxContextFrames: number;
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
    this.maxContextFrames = inferenceConfig.maxContextFrames;

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
    if (this.state.encoderState) {
      // Clean up old encoder state
      for (const { k, v } of this.state.encoderState.cachedKV) {
        this.backend.dispose(k);
        this.backend.dispose(v);
      }
      for (const s of this.state.encoderState.convStates) {
        this.backend.dispose(s);
      }
    }
    if (this.state.decoderState) {
      if (this.decoder instanceof RNNTGreedyDecoder) {
        this.decoder.disposeState(this.state.decoderState);
      } else {
        (this.decoder as TDTGreedyDecoder).disposeState(this.state.decoderState);
      }
    }
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

    // Extract one chunk
    const chunkAudio = this.state.audioBuffer.slice(0, this.chunkSizeSamples);
    this.state.audioBuffer = this.state.audioBuffer.slice(this.chunkSizeSamples);

    let audio = chunkAudio;
    if (this.resampler) {
      audio = this.resampler.resample(audio);
    }

    // Extract features
    const mel = this.featurePipeline.extractStreamingFeatures(audio);

    // Run encoder with cached state
    const { output: encoded, newState: newEncoderState } =
      this.encoder.forwardStreaming(mel, this.state.encoderState);

    this.backend.dispose(mel);

    // Update encoder state
    if (this.state.encoderState) {
      for (const { k, v } of this.state.encoderState.cachedKV) {
        this.backend.dispose(k);
        this.backend.dispose(v);
      }
      for (const s of this.state.encoderState.convStates) {
        this.backend.dispose(s);
      }
    }
    this.state.encoderState = newEncoderState;

    // Run decoder
    let newTokens: number[];
    if (this.decoder instanceof TDTGreedyDecoder) {
      const result = await this.decoder.decodeStreaming(
        encoded,
        this.state.decoderState,
        this.state.lastToken,
        this.state.tdtFrameOffset,
      );
      newTokens = result.tokens;
      if (this.state.decoderState) {
        this.decoder.disposeState(this.state.decoderState);
      }
      this.state.decoderState = result.newState;
      this.state.lastToken = result.newLastToken;
      this.state.tdtFrameOffset = result.newFrameOffset;
    } else {
      const rnntDecoder = this.decoder as RNNTGreedyDecoder;
      const result = await rnntDecoder.decodeStreaming(
        encoded,
        this.state.decoderState,
        this.state.lastToken,
      );
      newTokens = result.tokens;
      if (this.state.decoderState) {
        rnntDecoder.disposeState(this.state.decoderState);
      }
      this.state.decoderState = result.newState;
      this.state.lastToken = result.newLastToken;
    }

    this.backend.dispose(encoded);
    this.state.allTokens.push(...newTokens);
    this.state.totalAudioProcessed += chunkAudio.length;

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
    if (this.state.audioBuffer.length === 0) return null;

    const start = performance.now();
    let audio = this.state.audioBuffer;
    this.state.audioBuffer = new Float32Array(0);

    if (this.resampler) {
      audio = this.resampler.resample(audio);
    }

    // Pad to minimum feature extraction length if needed
    const minSamples = this.featurePipeline.frameLength;
    if (audio.length < minSamples) {
      const padded = new Float32Array(minSamples);
      padded.set(audio);
      audio = padded;
    }

    const mel = this.featurePipeline.extractStreamingFeatures(audio);
    const { output: encoded, newState: newEncoderState } =
      this.encoder.forwardStreaming(mel, this.state.encoderState);

    this.backend.dispose(mel);

    if (this.state.encoderState) {
      for (const { k, v } of this.state.encoderState.cachedKV) {
        this.backend.dispose(k);
        this.backend.dispose(v);
      }
      for (const s of this.state.encoderState.convStates) {
        this.backend.dispose(s);
      }
    }
    this.state.encoderState = newEncoderState;

    let newTokens: number[];
    if (this.decoder instanceof TDTGreedyDecoder) {
      const result = await this.decoder.decodeStreaming(
        encoded,
        this.state.decoderState,
        this.state.lastToken,
        this.state.tdtFrameOffset,
      );
      newTokens = result.tokens;
      if (this.state.decoderState) {
        this.decoder.disposeState(this.state.decoderState);
      }
      this.state.decoderState = result.newState;
      this.state.lastToken = result.newLastToken;
      this.state.tdtFrameOffset = result.newFrameOffset;
    } else {
      const rnntDecoder = this.decoder as RNNTGreedyDecoder;
      const result = await rnntDecoder.decodeStreaming(
        encoded,
        this.state.decoderState,
        this.state.lastToken,
      );
      newTokens = result.tokens;
      if (this.state.decoderState) {
        rnntDecoder.disposeState(this.state.decoderState);
      }
      this.state.decoderState = result.newState;
      this.state.lastToken = result.newLastToken;
    }

    this.backend.dispose(encoded);
    this.state.allTokens.push(...newTokens);

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
      encoderState: null,
      decoderState: null,
      lastToken: 0,
      tdtFrameOffset: 0,
      allTokens: [],
      totalAudioProcessed: 0,
    };
  }
}
