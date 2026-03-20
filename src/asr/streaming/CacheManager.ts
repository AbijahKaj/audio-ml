import type { ComputeBackend } from '../compute/index.js';
import type { FastConformerConfig } from '../model/ModelConfig.js';
import type { LSTMState } from '../decoder/PredictionNetwork.js';
import type { EncoderCache } from '../encoder/FastConformerEncoder.js';

/**
 * Streaming state cache for cache-aware FastConformer inference.
 *
 * Holds:
 *   - KV cache for each ConformerBlock attention layer (per-layer K and V tensors)
 *   - LSTM state for the prediction network
 *   - Last emitted token ID
 *   - Accumulated encoder frame count (for position tracking)
 *   - TDT frame offset (TDT may not consume all frames before the chunk boundary)
 *
 * Reference: Noroozi et al. (2023) — "Stateful Conformer with Cache-based
 * Inference for Streaming ASR"
 */
export interface StreamingCache {
  encoderCache: EncoderCache;
  predictionState: LSTMState;
  lastToken: number;
  encodedFrameCount: number;
  /** Partial tokens accumulated across chunks (not yet finalized). */
  partialTokens: number[];
  /** TDT: frame position within the latest encoder chunk that hasn't been consumed. */
  tdtFrameOffset: number;
}

export class CacheManager {
  constructor(
    private readonly backend: ComputeBackend,
    private readonly config: FastConformerConfig,
  ) {}

  /**
   * Create a fresh cache for the start of a new utterance.
   */
  create(): StreamingCache {
    const { encoderLayers, dModel, numHeads } = this.config;
    const headDim = dModel / numHeads;

    const encoderCache: EncoderCache = Array.from(
      { length: encoderLayers },
      () => ({
        k: this.backend.zeros([1, 0, numHeads, headDim]),
        v: this.backend.zeros([1, 0, numHeads, headDim]),
      }),
    );

    const predictionState: LSTMState = {
      h: this.backend.zeros([1, this.config.predHidden]),
      c: this.backend.zeros([1, this.config.predHidden]),
    };

    return {
      encoderCache,
      predictionState,
      lastToken: 0,
      encodedFrameCount: 0,
      partialTokens: [],
      tdtFrameOffset: 0,
    };
  }

  /**
   * Release all tensors held by the cache.
   */
  dispose(cache: StreamingCache): void {
    for (const block of cache.encoderCache) {
      this.backend.dispose(block.k);
      this.backend.dispose(block.v);
    }
    this.backend.dispose(cache.predictionState.h);
    this.backend.dispose(cache.predictionState.c);
  }

  /**
   * Reset cache to initial state (reuses existing tensors where possible).
   */
  reset(cache: StreamingCache): StreamingCache {
    this.dispose(cache);
    return this.create();
  }
}
