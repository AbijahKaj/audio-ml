import type { ComputeBackend } from '../compute/Backend';
import type { TensorHandle } from '../compute/types';
import type { FastConformerConfig } from '../model/ModelConfig';
import type { EncoderWeights } from '../model/types';
import { ConformerBlock } from './ConformerBlock';
import { ConvSubsampling } from './ConvSubsampling';
import { extendRelativePe } from './RelPositionalEncoding';

export class FastConformerEncoder {
  private readonly subsampling: ConvSubsampling;
  private readonly blocks: ConformerBlock[];

  constructor(
    private readonly backend: ComputeBackend,
    weights: EncoderWeights,
    private readonly config: FastConformerConfig,
  ) {
    const s = weights.subsampling;
    this.subsampling = new ConvSubsampling(
      backend,
      s.conv0,
      s.conv1,
      s.conv2,
      s.out,
    );
    this.blocks = weights.layers.map(
      w => new ConformerBlock(backend, w, config.dModel, config.numHeads),
    );
  }

  forward(melFeatures: TensorHandle): TensorHandle {
    let x = this.subsampling.forward(melFeatures);
    const sx = this.backend.getShape(x);
    const t = sx[1]!;

    if (this.config.xscale !== false) {
      const xs = this.backend.scale(x, Math.sqrt(this.config.dModel));
      this.backend.dispose(x);
      x = xs;
    }

    const posEmb = extendRelativePe(this.backend, t, this.config.dModel);
    for (const block of this.blocks) {
      const nx = block.forward(x, posEmb);
      this.backend.dispose(x);
      x = nx;
    }
    this.backend.dispose(posEmb);
    return x;
  }
}