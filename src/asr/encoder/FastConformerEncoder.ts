import type { ComputeBackend, TensorHandle } from '../compute/index.js';
import type { EncoderWeights } from '../model/WeightMapper.js';
import type { FastConformerConfig } from '../model/ModelConfig.js';
import { ConvSubsampling } from './ConvSubsampling.js';
import { ConformerBlock, type ConformerBlockCache } from './ConformerBlock.js';

export type EncoderCache = ConformerBlockCache[];

/**
 * FastConformer Encoder.
 *
 * Wraps:
 *   1. ConvSubsampling — reduces time by 8× and projects to d_model
 *   2. N ConformerBlocks (17 for Parakeet 120M, 24 for Nemotron 0.6B)
 *
 * Input:  mel features — flat Float32Array of [numFrames × numMelBands]
 * Output: encoded sequence — TensorHandle of shape [1, T/8, d_model]
 */
export class FastConformerEncoder {
  private readonly subsampling: ConvSubsampling;
  private readonly blocks: ConformerBlock[];
  private readonly config: FastConformerConfig;

  constructor(
    private readonly backend: ComputeBackend,
    weights: EncoderWeights,
    config: FastConformerConfig,
  ) {
    this.config = config;

    this.subsampling = new ConvSubsampling(
      backend,
      weights.subsampling,
      config.dModel,
      config.numMelBands,
      config.subsamplingConvChannels,
    );

    this.blocks = weights.layers.map(
      layerWeights => new ConformerBlock(
        backend,
        layerWeights,
        config.dModel,
        config.numHeads,
        config.convKernelSize,
      ),
    );
  }

  /**
   * Full (offline) forward pass.
   *
   * @param features  Flat Float32Array of shape [numFrames × numMelBands]
   * @param numFrames Number of time frames
   * @returns [1, T/8, d_model] encoder output
   */
  forward(features: Float32Array, numFrames: number): TensorHandle {
    const { numMelBands } = this.config;

    // Reshape flat features to [1, T, 80] tensor
    const melTensor = this.backend.tensor(features, [1, numFrames, numMelBands]);

    let x = this.subsampling.forward(melTensor);
    this.backend.dispose(melTensor);

    for (const block of this.blocks) {
      const next = block.forward(x);
      this.backend.dispose(x);
      x = next;
    }

    return x;
  }

  /**
   * Streaming forward pass for a single audio chunk.
   * Processes one chunk and returns updated KV cache.
   *
   * @param chunkFeatures Flat Float32Array for this chunk's mel features
   * @param chunkFrames   Number of time frames in this chunk
   * @param cache         Previous KV cache (null for first chunk)
   * @returns Encoded chunk output + updated cache
   */
  forwardStreaming(
    chunkFeatures: Float32Array,
    chunkFrames: number,
    cache: EncoderCache | null,
  ): { output: TensorHandle; cache: EncoderCache } {
    const { numMelBands } = this.config;
    const maxCacheLen = this.config.attContextSize[0];

    const melTensor = this.backend.tensor(
      chunkFeatures, [1, chunkFrames, numMelBands],
    );

    let x = this.subsampling.forward(melTensor);
    this.backend.dispose(melTensor);

    const newCache: EncoderCache = [];

    for (let i = 0; i < this.blocks.length; i++) {
      const blockCache = cache?.[i] ?? this.makeEmptyCache();
      const { output, cache: updatedCache } = this.blocks[i].forwardStreaming(
        x, blockCache, maxCacheLen,
      );
      this.backend.dispose(x);
      if (cache?.[i]) {
        this.backend.dispose(cache[i].k);
        this.backend.dispose(cache[i].v);
      }
      newCache.push(updatedCache);
      x = output;
    }

    return { output: x, cache: newCache };
  }

  private makeEmptyCache(): ConformerBlockCache {
    const { dModel, numHeads } = this.config;
    const headDim = dModel / numHeads;
    return {
      k: this.backend.zeros([1, 0, numHeads, headDim]),
      v: this.backend.zeros([1, 0, numHeads, headDim]),
    };
  }

  disposeCache(cache: EncoderCache): void {
    for (const c of cache) {
      this.backend.dispose(c.k);
      this.backend.dispose(c.v);
    }
  }
}
