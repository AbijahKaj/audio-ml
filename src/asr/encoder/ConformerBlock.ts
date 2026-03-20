import type { ComputeBackend } from '../compute/Backend';
import type { FastConformerConfig } from '../model/ModelConfig';
import type { ConformerLayerWeights } from '../model/weights';
import type { TensorHandle } from '../compute/types';
import { ConvModule } from './ConvModule';
import { FeedForward } from './FeedForward';
import { MultiHeadAttention } from './MultiHeadAttention';

export class ConformerBlock {
  private readonly ffn1: FeedForward;
  private readonly ffn2: FeedForward;
  private readonly attn: MultiHeadAttention;
  private readonly conv: ConvModule;

  constructor(
    backend: ComputeBackend,
    w: ConformerLayerWeights,
    config: FastConformerConfig,
  ) {
    this.ffn1 = new FeedForward(backend, w.feedForward1.linear1, w.feedForward1.linear2);
    this.ffn2 = new FeedForward(backend, w.feedForward2.linear1, w.feedForward2.linear2);
    this.attn = new MultiHeadAttention(backend, w.selfAtt, config);
    this.conv = new ConvModule(backend, w.conv);
    this.backend = backend;
    this.w = w;
  }

  private readonly backend: ComputeBackend;
  private readonly w: ConformerLayerWeights;

  forward(x: TensorHandle, posEmb: TensorHandle, padMask?: TensorHandle): TensorHandle {
    const b = this.backend;
    const w = this.w;
    let residual = x;

    let h = b.layerNorm(residual, w.normFeedForward1.weight, w.normFeedForward1.bias, 1e-5);
    h = this.ffn1.forward(h);
    residual = b.add(residual, b.scale(h, 0.5));
    b.dispose(h);

    h = b.layerNorm(residual, w.normSelfAtt.weight, w.normSelfAtt.bias, 1e-5);
    h = this.attn.forward(h, posEmb, padMask);
    residual = b.add(residual, h);
    b.dispose(h);

    h = b.layerNorm(residual, w.normConv.weight, w.normConv.bias, 1e-5);
    h = this.conv.forward(h);
    residual = b.add(residual, h);
    b.dispose(h);

    h = b.layerNorm(residual, w.normFeedForward2.weight, w.normFeedForward2.bias, 1e-5);
    h = this.ffn2.forward(h);
    residual = b.add(residual, b.scale(h, 0.5));
    b.dispose(h);

    const out = b.layerNorm(residual, w.normOut.weight, w.normOut.bias, 1e-5);
    b.dispose(residual);
    return out;
  }
}
