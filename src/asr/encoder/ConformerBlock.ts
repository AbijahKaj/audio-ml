import type { ComputeBackend, TensorHandle } from '../compute/index.js';
import { ComputeScope } from '../compute/index.js';
import type { ConformerBlockWeights } from '../model/WeightMapper.js';
import { FeedForward } from './FeedForward.js';
import { MultiHeadAttention } from './MultiHeadAttention.js';
import { ConvModule } from './ConvModule.js';

export interface ConformerBlockCache {
  k: TensorHandle;
  v: TensorHandle;
  convState?: TensorHandle;
}

/**
 * Full Conformer Block — the Macaron sandwich:
 *
 *   x = x + 0.5 * FFN1(x)
 *   x = x + MHSA(x)
 *   x = x + ConvModule(x)
 *   x = x + 0.5 * FFN2(x)
 *   x = LayerNorm(x)
 *
 * Reference: Gulati et al. (2020) — "Conformer: Convolution-augmented
 * Transformer for Speech Recognition".
 */
export class ConformerBlock {
  private readonly ffn1: FeedForward;
  private readonly attn: MultiHeadAttention;
  private readonly conv: ConvModule;
  private readonly ffn2: FeedForward;
  private readonly finalNorm: { weight: TensorHandle; bias: TensorHandle };

  constructor(
    private readonly backend: ComputeBackend,
    weights: ConformerBlockWeights,
    dModel: number,
    numHeads: number,
    convKernelSize: number,
  ) {
    this.ffn1 = new FeedForward(backend, weights.ffn1);
    this.attn = new MultiHeadAttention(backend, weights.attn, dModel, numHeads);
    this.conv = new ConvModule(backend, weights.conv, dModel, convKernelSize);
    this.ffn2 = new FeedForward(backend, weights.ffn2);
    this.finalNorm = weights.finalNorm;
  }

  /** Offline (full-context) forward pass. */
  forward(x: TensorHandle): TensorHandle {
    const scope = new ComputeScope();

    // FFN1 with scale 0.5
    const ffn1Out = scope.track(this.ffn1.forward(x));
    const ffn1Scaled = scope.track(this.backend.scale(ffn1Out, 0.5));
    let h = scope.track(this.backend.add(x, ffn1Scaled));

    // Self-attention
    const { output: attnOut } = this.attn.forward(h);
    const attnOutT = scope.track(attnOut);
    const hAfterAttn = scope.track(this.backend.add(h, attnOutT));
    h = hAfterAttn;

    // Conv module
    const convOut = scope.track(this.conv.forward(h));
    const hAfterConv = scope.track(this.backend.add(h, convOut));
    h = hAfterConv;

    // FFN2 with scale 0.5
    const ffn2Out = scope.track(this.ffn2.forward(h));
    const ffn2Scaled = scope.track(this.backend.scale(ffn2Out, 0.5));
    const hAfterFFN2 = scope.track(this.backend.add(h, ffn2Scaled));

    // Final layer norm
    const out = this.backend.layerNorm(
      hAfterFFN2,
      this.finalNorm.weight,
      this.finalNorm.bias,
      1e-5,
    );

    scope.dispose(this.backend);
    return out;
  }

  /**
   * Streaming forward pass — uses cached K, V from previous chunks.
   * Returns updated cache for the next chunk.
   */
  forwardStreaming(
    x: TensorHandle,
    cache: ConformerBlockCache,
    maxCacheLen: number,
  ): { output: TensorHandle; cache: ConformerBlockCache } {
    const scope = new ComputeScope();

    // FFN1 with scale 0.5
    const ffn1Out = scope.track(this.ffn1.forward(x));
    const ffn1Scaled = scope.track(this.backend.scale(ffn1Out, 0.5));
    let h = scope.track(this.backend.add(x, ffn1Scaled));

    // Self-attention with cached K, V
    const { output: attnOut, newK, newV } = this.attn.forward(
      h,
      cache.k,
      cache.v,
    );
    const attnOutT = scope.track(attnOut);
    let hAfterAttn = scope.track(this.backend.add(h, attnOutT));
    h = hAfterAttn;

    // Trim K, V cache to maxCacheLen
    const kvShape = this.backend.getShape(newK);
    const Tkv = kvShape[1] as number;
    let trimmedK: TensorHandle;
    let trimmedV: TensorHandle;
    if (Tkv > maxCacheLen) {
      const trimStart = Tkv - maxCacheLen;
      trimmedK = this.backend.slice(
        newK, [0, trimStart, 0], [-1, maxCacheLen, -1],
      );
      trimmedV = this.backend.slice(
        newV, [0, trimStart, 0], [-1, maxCacheLen, -1],
      );
      this.backend.dispose(newK);
      this.backend.dispose(newV);
    } else {
      trimmedK = newK;
      trimmedV = newV;
    }

    // Conv module
    const convOut = scope.track(this.conv.forward(h));
    const hAfterConv = scope.track(this.backend.add(h, convOut));
    h = hAfterConv;

    // FFN2 with scale 0.5
    const ffn2Out = scope.track(this.ffn2.forward(h));
    const ffn2Scaled = scope.track(this.backend.scale(ffn2Out, 0.5));
    const hAfterFFN2 = scope.track(this.backend.add(h, ffn2Scaled));

    // Final layer norm
    const out = this.backend.layerNorm(
      hAfterFFN2,
      this.finalNorm.weight,
      this.finalNorm.bias,
      1e-5,
    );

    scope.dispose(this.backend);
    return {
      output: out,
      cache: { k: trimmedK, v: trimmedV },
    };
  }
}
