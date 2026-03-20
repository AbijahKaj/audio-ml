import type { ComputeBackend } from '../compute/Backend';
import type { TensorHandle } from '../compute/types';
import type { DecoderState } from '../decoder/types';
import { TransducerDecoder } from '../decoder/TransducerDecoder';
import { FastConformerEncoder } from '../encoder/FastConformerEncoder';
import { FeaturePipeline } from '../features/FeaturePipeline';
import type { FastConformerConfig } from '../model/ModelConfig';
import { CacheManager, type StreamingCache } from './CacheManager';
import { Endpointer, type EndpointDecision } from './Endpointer';

export interface ChunkedInferenceResult {
  tokenIds: number[];
  endpoint: EndpointDecision;
  framesConsumed: number;
}

export class ChunkedInference {
  private cache: StreamingCache;

  constructor(
    private config: FastConformerConfig,
    private backend: ComputeBackend,
    private featurePipeline: FeaturePipeline,
    private encoder: FastConformerEncoder,
    private decoder: TransducerDecoder,
    private cacheManager: CacheManager,
    private endpointer?: Endpointer,
  ) {
    this.cache = this.cacheManager.create(config, backend, this.decoder.initialState());
  }

  async processFrame(pcmFrame: Float32Array): Promise<ChunkedInferenceResult> {
    const endpoint = this.endpointer ? this.endpointer.processFrame(pcmFrame) : 'speech';
    const melChunk = this.featurePipeline.extractStreamingFeatures(pcmFrame);
    if (!melChunk) {
      return {
        tokenIds: [],
        endpoint,
        framesConsumed: 0,
      };
    }

    const encoded = this.runEncoder(melChunk);
    const decoded = await this.decoder.decode(encoded.output, this.cache.predictionState);
    this.cacheManager.replaceEncoderCaches(this.cache, encoded.caches, this.backend);
    this.cacheManager.replaceDecoderState(this.cache, decoded.state, this.backend);
    this.cache.encodedFrameCount += decoded.framesConsumed;

    this.backend.dispose(melChunk);
    this.backend.dispose(encoded.output);

    if (endpoint === 'speech-end') {
      this.resetDecoderState();
    }

    return {
      tokenIds: decoded.tokenIds,
      endpoint,
      framesConsumed: decoded.framesConsumed,
    };
  }

  reset(): void {
    this.cacheManager.dispose(this.cache, this.backend);
    this.cache = this.cacheManager.create(this.config, this.backend, this.decoder.initialState());
    this.featurePipeline.reset();
    this.endpointer?.reset();
  }

  dispose(): void {
    this.cacheManager.dispose(this.cache, this.backend);
  }

  private runEncoder(melChunk: TensorHandle) {
    return this.encoder.forwardStreaming(melChunk, this.cache.encoderLayers);
  }

  private resetDecoderState(): void {
    const state: DecoderState = this.decoder.initialState();
    this.cacheManager.replaceDecoderState(this.cache, state, this.backend);
  }
}
