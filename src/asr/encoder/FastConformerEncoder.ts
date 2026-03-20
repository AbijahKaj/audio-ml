import type { ComputeBackend } from '../compute/Backend';
import type { FastConformerConfig } from '../model/ModelConfig';
import type { EncoderWeights } from '../model/weights';
import type { TensorHandle } from '../compute/types';
import { createExtendedRelativePositionalEncoding, createLocalRelativePositionalEncoding } from './posEmb';
import { ConformerBlock } from './ConformerBlock';
import { ConvSubsampling } from './ConvSubsampling';

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
    this.subsampling = new ConvSubsampling(backend, weights.subsampling);
    this.blocks = weights.layers.map(lw => new ConformerBlock(backend, lw, config));
  }

  forward(melFeatures: TensorHandle, padMask?: TensorHandle): TensorHandle {
    const b = this.backend;
    let x = this.subsampling.forward(melFeatures);
    const shape = b.getShape(x);
    const T = shape[1];
    if (this.config.xscale) {
      const scaled = b.scale(x, Math.sqrt(this.config.dModel));
      b.dispose(x);
      x = scaled;
    }

    let posEmb: TensorHandle;
    if (this.config.selfAttentionModel === 'rel_pos_local_attn') {
      const [left, right] = this.config.attContextSize;
      const pe = createLocalRelativePositionalEncoding(left, right, this.config.dModel);
      posEmb = b.tensor(pe, [1, left + right + 1, this.config.dModel]);
    } else if (this.config.selfAttentionModel === 'rel_pos') {
      const pe = createExtendedRelativePositionalEncoding(T, this.config.dModel);
      posEmb = b.tensor(pe, [1, 2 * T - 1, this.config.dModel]);
    } else {
      posEmb = b.zeros([1, T, this.config.dModel]);
    }

    for (const block of this.blocks) {
      const nx = block.forward(x, posEmb, padMask);
      b.dispose(x);
      x = nx;
    }

    b.dispose(posEmb);
    return x;
  }
}
