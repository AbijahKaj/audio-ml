import type { ComputeBackend } from '../compute/Backend';
import type { TensorHandle } from '../compute/types';
import type { FastConformerConfig } from '../model/ModelConfig';
import type { PredictionState } from '../decoder/PredictionNetwork';

export interface StreamingCache {
  attentionKV: Array<{ k: TensorHandle; v: TensorHandle }>;
  convStates: TensorHandle[];
  predictionState: PredictionState;
  lastToken: number;
  encodedFrameCount: number;
  tdtFrameOffset: number;
}

export class CacheManager {
  private backend: ComputeBackend;
  private config: FastConformerConfig;

  constructor(backend: ComputeBackend, config: FastConformerConfig) {
    this.backend = backend;
    this.config = config;
  }

  create(predHiddenSize: number, predNumLayers: number): StreamingCache {
    const headDim = this.config.dModel / this.config.numHeads;

    const attentionKV = Array.from({ length: this.config.encoderLayers }, () => ({
      k: this.backend.zeros([1, this.config.numHeads, 0, headDim]),
      v: this.backend.zeros([1, this.config.numHeads, 0, headDim]),
    }));

    const convStates = Array.from({ length: this.config.encoderLayers }, () =>
      this.backend.zeros([1, 0, this.config.dModel])
    );

    const predictionState: PredictionState = {
      h: Array.from({ length: predNumLayers }, () =>
        this.backend.zeros([1, predHiddenSize])
      ),
      c: Array.from({ length: predNumLayers }, () =>
        this.backend.zeros([1, predHiddenSize])
      ),
    };

    return {
      attentionKV,
      convStates,
      predictionState,
      lastToken: 0,
      encodedFrameCount: 0,
      tdtFrameOffset: 0,
    };
  }

  /**
   * Trim KV caches to maximum context window size to prevent unbounded growth.
   */
  trimCaches(cache: StreamingCache, maxContextFrames: number): StreamingCache {
    const newAttentionKV = cache.attentionKV.map(({ k, v }) => {
      const kLen = this.backend.getShape(k)[2] as number;
      if (kLen <= maxContextFrames) {
        return { k, v };
      }

      const trimStart = kLen - maxContextFrames;
      const heads = this.backend.getShape(k)[1] as number;
      const headDim = this.backend.getShape(k)[3] as number;

      const newK = this.backend.slice(k, [0, 0, trimStart, 0], [1, heads, maxContextFrames, headDim]);
      const newV = this.backend.slice(v, [0, 0, trimStart, 0], [1, heads, maxContextFrames, headDim]);

      this.backend.dispose(k);
      this.backend.dispose(v);

      return { k: newK, v: newV };
    });

    return {
      ...cache,
      attentionKV: newAttentionKV,
    };
  }

  dispose(cache: StreamingCache): void {
    for (const { k, v } of cache.attentionKV) {
      this.backend.dispose(k);
      this.backend.dispose(v);
    }
    for (const state of cache.convStates) {
      this.backend.dispose(state);
    }
    for (const h of cache.predictionState.h) {
      this.backend.dispose(h);
    }
    for (const c of cache.predictionState.c) {
      this.backend.dispose(c);
    }
  }
}
