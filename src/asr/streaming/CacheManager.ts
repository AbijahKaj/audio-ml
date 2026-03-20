import type { ComputeBackend } from '../compute/Backend';
import type { TensorHandle } from '../compute/types';
import type { DecoderState } from '../decoder/types';
import type { FastConformerConfig } from '../model/ModelConfig';
import type { ConformerLayerCache } from '../encoder/types';

export interface StreamingCache {
  attentionKV: Array<{ k: TensorHandle; v: TensorHandle } | undefined>;
  convStates: Array<TensorHandle | undefined>;
  encoderLayers: ConformerLayerCache[];
  predictionState: DecoderState;
  lastToken: number;
  encodedFrameCount: number;
  tdtFrameOffset: number;
}

export class CacheManager {
  create(
    config: FastConformerConfig,
    backend: ComputeBackend,
    predictionState?: DecoderState,
  ): StreamingCache {
    const decoderState = predictionState ?? this.createDecoderState(config, backend);
    return {
      attentionKV: Array.from({ length: config.encoderLayers }, () => undefined),
      convStates: Array.from({ length: config.encoderLayers }, () => undefined),
      encoderLayers: Array.from({ length: config.encoderLayers }, () => ({})),
      predictionState: decoderState,
      lastToken: decoderState.lastToken,
      encodedFrameCount: 0,
      tdtFrameOffset: decoderState.frameOffset,
    };
  }

  dispose(cache: StreamingCache, backend: ComputeBackend): void {
    for (const attention of cache.attentionKV) {
      if (attention?.k) backend.dispose(attention.k);
      if (attention?.v) backend.dispose(attention.v);
    }
    for (const convState of cache.convStates) {
      if (convState) backend.dispose(convState);
    }

    this.disposeDecoderState(cache.predictionState, backend);
  }

  replaceEncoderCaches(
    cache: StreamingCache,
    nextLayers: ConformerLayerCache[],
    backend: ComputeBackend,
  ): void {
    for (let i = 0; i < cache.encoderLayers.length; i++) {
      const previous = cache.encoderLayers[i];
      const next = nextLayers[i] ?? {};

      if (previous.attention?.k && previous.attention.k !== next.attention?.k) {
        backend.dispose(previous.attention.k);
      }
      if (previous.attention?.v && previous.attention.v !== next.attention?.v) {
        backend.dispose(previous.attention.v);
      }
      if (previous.convState && previous.convState !== next.convState) {
        backend.dispose(previous.convState);
      }

      cache.encoderLayers[i] = next;
      cache.attentionKV[i] = next.attention;
      cache.convStates[i] = next.convState;
    }
  }

  replaceDecoderState(cache: StreamingCache, nextState: DecoderState, backend: ComputeBackend): void {
    if (cache.predictionState.h !== nextState.h) {
      backend.dispose(cache.predictionState.h);
    }
    if (cache.predictionState.c !== nextState.c) {
      backend.dispose(cache.predictionState.c);
    }
    cache.predictionState = nextState;
    cache.lastToken = nextState.lastToken;
    cache.tdtFrameOffset = nextState.frameOffset;
  }

  private createDecoderState(config: FastConformerConfig, backend: ComputeBackend): DecoderState {
    return {
      h: backend.zeros([1, config.predHidden]),
      c: backend.zeros([1, config.predHidden]),
      lastToken: config.blankId,
      frameOffset: 0,
    };
  }

  private disposeDecoderState(state: DecoderState, backend: ComputeBackend): void {
    backend.dispose(state.h);
    backend.dispose(state.c);
  }
}
