import type { ComputeBackend } from '../compute/Backend';
import type { FastConformerConfig } from '../model/ModelConfig';
import type { TensorHandle } from '../compute/types';

/** Per-layer streaming cache placeholders (KV + conv states). */
export interface StreamingCache {
  attentionKV: Array<{ k: TensorHandle; v: TensorHandle }>;
  convStates: TensorHandle[];
  predictionState: { h: TensorHandle; c: TensorHandle };
  lastToken: number;
  encodedFrameCount: number;
  tdtFrameOffset: number;
}

export class CacheManager {
  create(_config: FastConformerConfig, backend: ComputeBackend): StreamingCache {
    return {
      attentionKV: [],
      convStates: [],
      predictionState: {
        h: backend.zeros([1, 1]),
        c: backend.zeros([1, 1]),
      },
      lastToken: 0,
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
    backend.dispose(cache.predictionState.h);
    backend.dispose(cache.predictionState.c);
  }
}