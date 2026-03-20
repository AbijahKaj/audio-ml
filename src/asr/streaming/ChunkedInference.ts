import type { ComputeBackend } from '../compute/index.js';
import type { FastConformerConfig } from '../model/ModelConfig.js';
import type { FastConformerEncoder } from '../encoder/FastConformerEncoder.js';
import type { RNNTGreedyDecoder } from '../decoder/RNNTGreedyDecoder.js';
import type { TDTGreedyDecoder } from '../decoder/TDTGreedyDecoder.js';
import type { SentencePieceDecoder } from '../text/SentencePieceDecoder.js';
import type { FeaturePipeline } from '../features/FeaturePipeline.js';
import { CacheManager, type StreamingCache } from './CacheManager.js';

export interface ChunkResult {
  partialText: string;
  newTokens: number[];
  isFinal: boolean;
}

/**
 * Chunked streaming inference engine.
 *
 * Processes audio in fixed-size chunks, maintaining KV/LSTM state across
 * chunks to produce incremental transcription results.
 *
 * Usage:
 *   const engine = new ChunkedInference(encoder, decoder, pipeline, tokenizer, config);
 *   const cache = engine.startUtterance();
 *   for each audio chunk:
 *     const result = await engine.processChunk(chunk, cache);
 *     display result.partialText;
 *   const final = await engine.finalizeUtterance(cache);
 *
 * Reference: Noroozi et al. (2023) — "Stateful Conformer with Cache-based
 * Inference for Streaming ASR"
 */
export class ChunkedInference {
  private readonly cacheManager: CacheManager;

  constructor(
    private readonly backend: ComputeBackend,
    private readonly encoder: FastConformerEncoder,
    private readonly decoder: RNNTGreedyDecoder | TDTGreedyDecoder,
    private readonly featurePipeline: FeaturePipeline,
    private readonly tokenizer: SentencePieceDecoder,
    config: FastConformerConfig,
  ) {
    this.cacheManager = new CacheManager(backend, config);
  }

  startUtterance(): StreamingCache {
    return this.cacheManager.create();
  }

  /**
   * Process one audio chunk in streaming mode.
   *
   * @param audioChunk  16kHz PCM samples for this chunk
   * @param cache       Mutable streaming cache (updated in-place)
   * @returns           Partial text and new token IDs from this chunk
   */
  async processChunk(
    audioChunk: Float32Array,
    cache: StreamingCache,
  ): Promise<ChunkResult> {
    // Extract features for this chunk
    const { features, numFrames } = this.featurePipeline.extractFeatures(audioChunk);

    if (numFrames === 0) {
      return { partialText: this.tokenizer.decode(cache.partialTokens), newTokens: [], isFinal: false };
    }

    // Run streaming encoder (uses cached KV from previous chunks)
    const { output: encoderOut, cache: newEncoderCache } =
      this.encoder.forwardStreaming(features, numFrames, cache.encoderCache);

    // Dispose old encoder cache
    for (const block of cache.encoderCache) {
      this.backend.dispose(block.k);
      this.backend.dispose(block.v);
    }
    cache.encoderCache = newEncoderCache;
    cache.encodedFrameCount += this.backend.getShape(encoderOut)[1] as number;

    // Decode new encoder frames
    const newTokens = await this.decoder.decode(encoderOut);
    this.backend.dispose(encoderOut);

    cache.partialTokens.push(...newTokens);
    const partialText = this.tokenizer.decode(cache.partialTokens);

    return { partialText, newTokens, isFinal: false };
  }

  /**
   * Finalize the utterance — returns the complete transcript and resets cache.
   */
  finalizeUtterance(cache: StreamingCache): { text: string; tokenIds: number[] } {
    const text = this.tokenizer.decode(cache.partialTokens);
    const tokenIds = [...cache.partialTokens];
    return { text, tokenIds };
  }

  disposeCache(cache: StreamingCache): void {
    this.cacheManager.dispose(cache);
  }
}
