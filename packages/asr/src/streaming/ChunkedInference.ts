import type { ComputeBackend } from '../compute/Backend';
import type { FastConformerConfig } from '../model/ModelConfig';
import { FeaturePipeline } from '../features/FeaturePipeline';
import { FastConformerEncoder, type StreamingEncoderState } from '../encoder/FastConformerEncoder';
import type { TDTGreedyDecoder } from '../decoder/TDTGreedyDecoder';
import type { PredictionState } from '../decoder/PredictionNetwork';
import { SentencePieceDecoder } from '../text/SentencePieceDecoder';
import { Resampler } from '../features/Resampler';

export interface StreamingResult {
  text: string;
  isFinal: boolean;
  isPartial: boolean;
  latencyMs: number;
  decoderType: 'tdt';
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
}

/**
 * Cache-aware chunked streaming inference engine.
 *
 * Each chunk only runs the encoder on NEW frames (using cached KV + conv
 * states from previous chunks). The decoder continues from its previous
 * LSTM state + last emitted token, so no work is repeated.
 *
 * Complexity is O(chunk_size) per step, not O(total_audio) like the naive
 * re-encode approach.
 */
export class ChunkedInference {
  private backend: ComputeBackend;
  private config: FastConformerConfig;
  private featurePipeline: FeaturePipeline;
  private encoder: FastConformerEncoder;
  private decoder: TDTGreedyDecoder;
  private tokenizer: SentencePieceDecoder;
  private resampler: Resampler | null;

  private chunkSizeSamples: number;
  private state: StreamingState;
  private processingLock: Promise<void> = Promise.resolve();

  constructor(
    backend: ComputeBackend,
    config: FastConformerConfig,
    featurePipeline: FeaturePipeline,
    encoder: FastConformerEncoder,
    decoder: TDTGreedyDecoder,
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

  async feedAudio(pcm: Float32Array): Promise<StreamingResult | null> {
    const newBuffer = new Float32Array(this.state.audioBuffer.length + pcm.length);
    newBuffer.set(this.state.audioBuffer);
    newBuffer.set(pcm, this.state.audioBuffer.length);
    this.state.audioBuffer = newBuffer;

    if (this.state.audioBuffer.length < this.chunkSizeSamples) {
      return null;
    }

    // Serialize chunk processing to prevent concurrent access to shared
    // encoder/decoder state. Without this, overlapping async processChunk
    // calls can dispose tensors that another call still references.
    let result: StreamingResult | null = null;
    const prev = this.processingLock;
    this.processingLock = prev
      .then(() => this.processChunk())
      .then(r => { result = r; });
    await this.processingLock;
    return result;
  }

  async flush(): Promise<StreamingResult> {
    // Wait for any in-flight processChunk to complete before flushing
    await this.processingLock;

    if (this.state.audioBuffer.length > 0) {
      const result = await this.processRemaining();
      if (result) return { ...result, isFinal: true };
    }

    return {
      text: this.tokenizer.decode(this.state.allTokens),
      isFinal: true,
      isPartial: false,
      latencyMs: 0,
      decoderType: this.config.decoderType,
    };
  }

  reset(): void {
    this.disposeState();
    this.state = this.createInitialState();
    this.processingLock = Promise.resolve();
    this.featurePipeline.resetStreamingState();
  }

  get currentText(): string {
    return this.tokenizer.decode(this.state.allTokens);
  }

  get tokenCount(): number {
    return this.state.allTokens.length;
  }

  private async processChunk(): Promise<StreamingResult | null> {
    const start = performance.now();

    const chunkAudio = this.state.audioBuffer.subarray(0, this.chunkSizeSamples);
    this.state.audioBuffer = this.state.audioBuffer.subarray(this.chunkSizeSamples);

    const { encoded, oldEncoderState } = this.encodeChunk(chunkAudio);
    const newTokens = await this.decodeIncremental(encoded);

    // Dispose AFTER decoding: getData() inside the decoder flushes the
    // GPU command queue, so by this point all encoder GPU operations that
    // referenced old state buffers have completed.
    this.backend.dispose(encoded);
    this.disposeEncoderState(oldEncoderState);

    if (newTokens.length > 0) {
      this.state.allTokens.push(...newTokens);
    }

    return {
      text: this.tokenizer.decode(this.state.allTokens),
      isFinal: false,
      isPartial: true,
      latencyMs: performance.now() - start,
      decoderType: this.config.decoderType,
    };
  }

  private async processRemaining(): Promise<StreamingResult | null> {
    const start = performance.now();

    let audio = new Float32Array(this.state.audioBuffer);
    this.state.audioBuffer = new Float32Array(0);

    if (audio.length < this.featurePipeline.frameLength) {
      const padded = new Float32Array(this.featurePipeline.frameLength);
      padded.set(audio);
      audio = padded;
    }

    const { encoded, oldEncoderState } = this.encodeChunk(audio);
    const newTokens = await this.decodeIncremental(encoded);

    this.backend.dispose(encoded);
    this.disposeEncoderState(oldEncoderState);

    if (newTokens.length > 0) {
      this.state.allTokens.push(...newTokens);
    }

    return {
      text: this.tokenizer.decode(this.state.allTokens),
      isFinal: true,
      isPartial: false,
      latencyMs: performance.now() - start,
      decoderType: this.config.decoderType,
    };
  }

  /**
   * Extract mel features from a chunk and run the encoder with cached state.
   * Returns both the encoded output and the old encoder state for deferred
   * disposal (old state must not be freed until GPU operations complete).
   */
  private encodeChunk(chunkPcm: Float32Array): {
    encoded: TensorHandle;
    oldEncoderState: StreamingEncoderState | null;
  } {
    let audio = chunkPcm;
    if (this.resampler) {
      audio = this.resampler.resample(audio);
    }

    const mel = this.featurePipeline.extractStreamingFeatures(audio);
    const oldEncoderState = this.state.encoderState;
    const { output, newState } = this.encoder.forwardStreaming(mel, oldEncoderState);
    this.backend.dispose(mel);

    this.state.encoderState = newState;
    return { encoded: output, oldEncoderState };
  }

  private disposeEncoderState(state: StreamingEncoderState | null): void {
    if (!state) return;
    for (const { k, v } of state.cachedKV) {
      this.backend.dispose(k);
      this.backend.dispose(v);
    }
    for (const s of state.convStates) {
      this.backend.dispose(s);
    }
  }

  /**
   * Run the decoder only on new encoder frames, continuing from
   * the previous prediction state and last emitted token.
   */
  private async decodeIncremental(encoderOutput: TensorHandle): Promise<number[]> {
    const T = this.backend.getShape(encoderOutput)[1] as number;
    if (T === 0) return [];

    const result = await this.decoder.decodeStreaming(
      encoderOutput,
      this.state.decoderState,
      this.state.lastToken,
      this.state.tdtFrameOffset,
    );
    this.state.decoderState = result.newState;
    this.state.lastToken = result.newLastToken;
    this.state.tdtFrameOffset = result.newFrameOffset;
    return result.tokens;
  }

  private disposeState(): void {
    if (this.state.decoderState) {
      for (const h of this.state.decoderState.h) this.backend.dispose(h);
      for (const c of this.state.decoderState.c) this.backend.dispose(c);
    }
    if (this.state.encoderState) {
      for (const { k, v } of this.state.encoderState.cachedKV) {
        this.backend.dispose(k);
        this.backend.dispose(v);
      }
      for (const s of this.state.encoderState.convStates) {
        this.backend.dispose(s);
      }
    }
  }

  private createInitialState(): StreamingState {
    return {
      audioBuffer: new Float32Array(0),
      encoderState: null,
      decoderState: null,
      lastToken: this.config.vocabSize - 1, // blank token (NeMo convention)
      tdtFrameOffset: 0,
      allTokens: [],
    };
  }
}

type TensorHandle = import('../compute/types').TensorHandle;
