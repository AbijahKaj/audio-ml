import type { ComputeBackend } from '../compute/Backend';
import type { TensorHandle } from '../compute/types';
import type { FastConformerConfig } from '../model/ModelConfig';
import type { EncoderWeights } from '../model/WeightMapper';
import { ConformerBlock } from './ConformerBlock';
import { ConvSubsampling } from './ConvSubsampling';

export class FastConformerEncoder {
  private readonly subsampling: ConvSubsampling;
  private readonly blocks: ConformerBlock[];

  constructor(
    private readonly backend: ComputeBackend,
    weights: EncoderWeights,
    private readonly config: FastConformerConfig,
  ) {
    this.subsampling = new ConvSubsampling(backend, weights, config);
    this.blocks = weights.layers.map(
      (layerWeights) => new ConformerBlock(backend, layerWeights, config.numHeads, config.convKernelSize),
    );
  }

  forward(melFeatures: TensorHandle): TensorHandle {
    let state = this.subsampling.forward(melFeatures);

    for (const block of this.blocks) {
      state = block.forward(state);
    }

    return state;
  }

  getChunkFrames(ms: number): number {
    const inputFrames = Math.max(1, Math.round(ms / this.config.hopSizeMs));
    return Math.max(1, Math.ceil(inputFrames / this.config.subsamplingFactor));
  }

  getOutputShape(input: TensorHandle): readonly number[] {
    return this.backend.getShape(input);
  }
}
