import { ComputeScope } from '../compute/ComputeScope';
import type { ComputeBackend } from '../compute/Backend';
import type { TensorHandle } from '../compute/types';
import type { ConformerLayerWeights } from '../model/ModelWeights';
import type { ConformerLayerCache } from './types';
import { ConvModule } from './ConvModule';
import { FeedForward } from './FeedForward';
import { MultiHeadAttention } from './MultiHeadAttention';

export interface StreamingBlockOutput {
  output: TensorHandle;
  cache: ConformerLayerCache;
}

export class ConformerBlock {
  private readonly ffn1: FeedForward;
  private readonly ffn2: FeedForward;
  private readonly attn: MultiHeadAttention;
  private readonly conv: ConvModule;

  constructor(
    private backend: ComputeBackend,
    private weights: ConformerLayerWeights,
    numHeads: number,
    contextLeft: number,
  ) {
    this.ffn1 = new FeedForward(backend, weights.ffn1);
    this.ffn2 = new FeedForward(backend, weights.ffn2);
    this.attn = new MultiHeadAttention(backend, weights.attn, numHeads, contextLeft);
    this.conv = new ConvModule(backend, weights.conv);
  }

  forward(x: TensorHandle): TensorHandle {
    const scope = new ComputeScope();
    const ff1 = scope.track(this.ffn1.forward(x));
    const x1 = scope.track(this.backend.add(x, this.backend.scale(ff1, 0.5)));
    const attn = scope.track(this.attn.forward(x1));
    const x2 = scope.track(this.backend.add(x1, attn));
    const conv = scope.track(this.conv.forward(x2));
    const x3 = scope.track(this.backend.add(x2, conv));
    const ff2 = scope.track(this.ffn2.forward(x3));
    const x4 = scope.track(this.backend.add(x3, this.backend.scale(ff2, 0.5)));
    const output = this.backend.layerNorm(x4, this.weights.finalNorm.weight, this.weights.finalNorm.bias, 1e-5);
    scope.keep(output);
    scope.dispose(this.backend);
    return output;
  }

  forwardStreaming(chunk: TensorHandle, cache?: ConformerLayerCache): StreamingBlockOutput {
    const scope = new ComputeScope();
    const ff1 = scope.track(this.ffn1.forward(chunk));
    const x1 = scope.track(this.backend.add(chunk, this.backend.scale(ff1, 0.5)));

    const attnResult = this.attn.forwardStreaming(x1, cache?.attention);
    const attnOut = scope.track(attnResult.output);
    const x2 = scope.track(this.backend.add(x1, attnOut));

    const convResult = this.conv.forwardStreaming(x2, cache?.convState);
    const convOut = scope.track(convResult.output);
    const x3 = scope.track(this.backend.add(x2, convOut));

    const ff2 = scope.track(this.ffn2.forward(x3));
    const x4 = scope.track(this.backend.add(x3, this.backend.scale(ff2, 0.5)));
    const output = this.backend.layerNorm(
      x4,
      this.weights.finalNorm.weight,
      this.weights.finalNorm.bias,
      1e-5,
    );

    scope.keep(output);
    scope.keep(attnResult.cache.k);
    scope.keep(attnResult.cache.v);
    scope.keep(convResult.convState);
    scope.dispose(this.backend);

    return {
      output,
      cache: {
        attention: attnResult.cache,
        convState: convResult.convState,
      },
    };
  }
}
