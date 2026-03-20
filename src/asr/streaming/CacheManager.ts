import type { ComputeBackend } from '../compute/Backend';
import type { FastConformerConfig } from '../model/ModelConfig';
import type { TensorHandle } from '../compute/types';

/**
 * Cache-aware streaming state (attention KV + conv caches + decoder state).
 * Encoder cache tensors are populated by a full cache-aware encoder forward (future work).
 */
export interface StreamingCache {
  attentionKV: Array<{ k: TensorHandle; v: TensorHandle }>;
  convStates: TensorHandle[];
  predictionState: { h: TensorHandle; c: TensorHandle } | null;
  lastToken: number;
  encodedFrameCount: number;
  tdtFrameOffset: number;
}

export class CacheManager {
  create(config: FastConformerConfig, backend: ComputeBackend): StreamingCache {
    const layers: Array<{ k: TensorHandle; v: TensorHandle }> = [];
    for (let i = 0; i < config.encoderLayers; i++) {
      layers.push({
        k: backend.zeros([1, 1, config.dModel]),
        v: backend.zeros([1, 1, config.dModel]),
      });
    }
    return {
      attentionKV: layers,
      convStates: [],
      predictionState: null,
      lastToken: config.blankTokenId,
      encodedFrameCount: 0,
      tdtFrameOffset: 0,
    };
  }

  dispose(cache: StreamingCache, backend: ComputeBackend): void {
    for (const { k, v } of cache.attentionKV) {
      backend.dispose(k);
      backend.dispose(v);
    }
    for (const t of cache.convStates) {
      backend.dispose(t);
    }
    if (cache.predictionState) {
      backend.dispose(cache.predictionState.h);
      backend.dispose(cache.predictionState.c);
    }
  }
}
