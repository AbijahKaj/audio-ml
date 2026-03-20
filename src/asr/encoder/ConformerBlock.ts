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
    let h = this.backend.add(x, this.backend.scale(this.ffn1.forward(x), 0.5));

    const attnResult = this.attn.forwardStreaming(h, cachedK, cachedV, mask);
    h = this.backend.add(h, attnResult.output);

    const convResult = this.conv.forwardStreaming(h, convState);
    h = this.backend.add(h, convResult.output);

    h = this.backend.add(h, this.backend.scale(this.ffn2.forward(h), 0.5));
    h = this.backend.layerNorm(h, this.finalNormWeight, this.finalNormBias, 1e-5);

    return {
      output: h,
      newK: attnResult.newK,
      newV: attnResult.newV,
      newConvState: convResult.newConvState,
    };
  }
}
