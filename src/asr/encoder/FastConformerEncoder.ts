import type { ComputeBackend } from '../compute/Backend';
import type { TensorHandle } from '../compute/types';
import type { FastConformerConfig } from '../model/ModelConfig';
import type { EncoderWeights } from '../model/ModelWeights';
import { ConformerBlock } from './ConformerBlock';
import { ConvSubsampling } from './ConvSubsampling';
import type { ConformerLayerCache } from './types';

export interface StreamingEncoderOutput {
  output: TensorHandle;
  caches: ConformerLayerCache[];
}

export class FastConformerEncoder {
  private readonly subsampling: ConvSubsampling;
  private readonly blocks: ConformerBlock[];

  constructor(
    private backend: ComputeBackend,
    private weights: EncoderWeights,
    private config: FastConformerConfig,
  ) {
    this.subsampling = new ConvSubsampling(backend, weights.subsampling, config.subsamplingFactor);
    this.blocks = weights.layers.map(
      (layer) => new ConformerBlock(backend, layer, config.numHeads, config.attContextSize[0]),
    );
  }

  forward(melFeatures: TensorHandle): TensorHandle {
    let x = this.subsampling.forward(melFeatures);
    for (const block of this.blocks) {
      const next = block.forward(x);
      this.backend.dispose(x);
      x = next;
    }
    return x;
  }

  forwardStreaming(
    melChunk: TensorHandle,
    caches: ConformerLayerCache[] = [],
  ): StreamingEncoderOutput {
    let x = this.subsampling.forward(melChunk);
    const nextCaches: ConformerLayerCache[] = [];

    for (let i = 0; i < this.blocks.length; i++) {
      const result = this.blocks[i].forwardStreaming(x, caches[i]);
      this.backend.dispose(x);
      x = result.output;
      nextCaches.push(result.cache);
    }

    return {
      output: x,
      caches: nextCaches,
    };
  }

  getLayerCount(): number {
    return this.config.encoderLayers;
  }

  getSubsamplingFactor(): number {
    return this.config.subsamplingFactor;
  }
}
