import type { ComputeBackend } from '../compute/Backend';
import type { TensorHandle } from '../compute/types';
import type { ConformerBlockWeights } from '../model/WeightMapper';
import type { FastConformerConfig } from '../model/ModelConfig';
import { FeedForward } from './FeedForward';
import { MultiHeadAttention } from './MultiHeadAttention';
import { ConvModule } from './ConvModule';

export class ConformerBlock {
  private backend: ComputeBackend;
  private ffn1: FeedForward;
  private attn: MultiHeadAttention;
  private conv: ConvModule;
  private ffn2: FeedForward;
  private finalNormWeight: TensorHandle;
  private finalNormBias: TensorHandle;

  constructor(backend: ComputeBackend, weights: ConformerBlockWeights, config: FastConformerConfig) {
    this.backend = backend;
    this.ffn1 = new FeedForward(backend, weights.ffn1);
    this.attn = new MultiHeadAttention(backend, weights.attn, config.numHeads, config.dModel);
    this.conv = new ConvModule(backend, weights.conv, config.convKernelSize);
    this.ffn2 = new FeedForward(backend, weights.ffn2);
    this.finalNormWeight = weights.finalNorm.weight;
    this.finalNormBias = weights.finalNorm.bias;
  }

  forward(x: TensorHandle, mask?: TensorHandle): TensorHandle {
    return this.backend.tidy(() => {
      // Macaron-style sandwich: FFN -> MHSA -> Conv -> FFN -> LayerNorm
      let h = this.backend.add(x, this.backend.scale(this.ffn1.forward(x), 0.5));
      h = this.backend.add(h, this.attn.forward(h, mask));
      h = this.backend.add(h, this.conv.forward(h));
      h = this.backend.add(h, this.backend.scale(this.ffn2.forward(h), 0.5));
      h = this.backend.layerNorm(h, this.finalNormWeight, this.finalNormBias, 1e-5);
      return h;
    });
  }

  forwardStreaming(
    x: TensorHandle,
    cachedK: TensorHandle | null,
    cachedV: TensorHandle | null,
    convState: TensorHandle | null,
    mask?: TensorHandle,
  ): {
    output: TensorHandle;
    newK: TensorHandle;
    newV: TensorHandle;
    newConvState: TensorHandle;
  } {
    const temps: TensorHandle[] = [];

    const ffn1Out = this.ffn1.forward(x);
    const ffn1Scaled = this.backend.scale(ffn1Out, 0.5);
    temps.push(ffn1Out, ffn1Scaled);
    let h = this.backend.add(x, ffn1Scaled);
    temps.push(h);

    const attnResult = this.attn.forwardStreaming(h, cachedK, cachedV, mask);
    temps.push(attnResult.output);
    const h2 = this.backend.add(h, attnResult.output);
    temps.push(h2);
    h = h2;

    const convResult = this.conv.forwardStreaming(h, convState);
    temps.push(convResult.output);
    const h3 = this.backend.add(h, convResult.output);
    temps.push(h3);
    h = h3;

    const ffn2Out = this.ffn2.forward(h);
    const ffn2Scaled = this.backend.scale(ffn2Out, 0.5);
    temps.push(ffn2Out, ffn2Scaled);
    const h4 = this.backend.add(h, ffn2Scaled);
    temps.push(h4);

    const output = this.backend.layerNorm(h4, this.finalNormWeight, this.finalNormBias, 1e-5);

    for (const t of temps) this.backend.dispose(t);

    return {
      output,
      newK: attnResult.newK,
      newV: attnResult.newV,
      newConvState: convResult.newConvState,
    };
  }
}
