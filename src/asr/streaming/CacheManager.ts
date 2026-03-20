import type { ComputeBackend } from '../compute/Backend.js';
import type { TensorHandle } from '../compute/types.js';
import type { FastConformerConfig } from '../model/ModelConfig.js';
import type { PredictionState } from '../decoder/PredictionNetwork.js';

export interface StreamingCache {
  attentionKV: Array<{ k: TensorHandle | null; v: TensorHandle | null }>;
  convStates: Array<TensorHandle | null>;
  predictionState: PredictionState | null;
  lastToken: number;
  encodedFrameCount: number;
  tdtFrameOffset: number;
}

export class CacheManager {
  create(config: FastConformerConfig): StreamingCache {
    return {
      attentionKV: Array.from({ length: config.encoderLayers }, () => ({ k: null, v: null })),
      convStates: Array.from({ length: config.encoderLayers }, () => null),
      predictionState: null,
      lastToken: 0,
      encodedFrameCount: 0,
      tdtFrameOffset: 0,
    };
  }

  dispose(cache: StreamingCache, backend: ComputeBackend): void {
    for (const layer of cache.attentionKV) {
      if (layer.k) {
        backend.dispose(layer.k);
      }
      if (layer.v) {
        backend.dispose(layer.v);
      }
    }

    for (const state of cache.convStates) {
      if (state) {
        backend.dispose(state);
      }
    }

    if (cache.predictionState) {
      backend.dispose(cache.predictionState.h);
      backend.dispose(cache.predictionState.c);
    }
  }
}
