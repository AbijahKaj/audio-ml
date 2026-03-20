import { ComputeScope } from '../compute/ComputeScope';
import type { ComputeBackend } from '../compute/Backend';
import type { TensorHandle } from '../compute/types';
import type { AttentionWeights } from '../model/ModelWeights';
import type { AttentionCache } from './types';
import { Linear } from './Linear';
import { RelativePositionalEncoding } from './RelativePositionalEncoding';

export interface StreamingAttentionOutput {
  output: TensorHandle;
  cache: AttentionCache;
}

export class MultiHeadAttention {
  private readonly qProj: Linear;
  private readonly kProj: Linear;
  private readonly vProj: Linear;
  private readonly outProj: Linear;
  private readonly headDim: number;
  private readonly posEncoding: RelativePositionalEncoding;

  constructor(
    private backend: ComputeBackend,
    private weights: AttentionWeights,
    private numHeads: number,
    private contextLeft: number,
  ) {
    this.qProj = new Linear(backend, weights.q);
    this.kProj = new Linear(backend, weights.k);
    this.vProj = new Linear(backend, weights.v);
    this.outProj = new Linear(backend, weights.out);
    const outDim = this.backend.getShape(weights.q.weight)[0];
    this.headDim = Math.floor(outDim / numHeads);
    this.posEncoding = new RelativePositionalEncoding(
      backend,
      numHeads,
      this.headDim,
      weights.posBiasU,
      weights.posBiasV,
    );
  }

  forward(x: TensorHandle, mask?: TensorHandle): TensorHandle {
    const scope = new ComputeScope();
    const normed = scope.track(this.backend.layerNorm(x, this.weights.norm.weight, this.weights.norm.bias, 1e-5));
    const q = scope.track(this.splitHeads(this.qProj.forward(normed)));
    const k = scope.track(this.splitHeads(this.kProj.forward(normed)));
    const v = scope.track(this.splitHeads(this.vProj.forward(normed)));
    const output = this.attend(q, k, v, mask);
    scope.keep(output);
    scope.dispose(this.backend);
    return output;
  }

  forwardStreaming(chunk: TensorHandle, cache?: AttentionCache, mask?: TensorHandle): StreamingAttentionOutput {
    const scope = new ComputeScope();
    const normed = scope.track(
      this.backend.layerNorm(chunk, this.weights.norm.weight, this.weights.norm.bias, 1e-5),
    );
    const q = scope.track(this.splitHeads(this.qProj.forward(normed)));
    const chunkK = scope.track(this.splitHeads(this.kProj.forward(normed)));
    const chunkV = scope.track(this.splitHeads(this.vProj.forward(normed)));

    const fullK = cache ? scope.track(this.backend.concat([cache.k, chunkK], 2)) : chunkK;
    const fullV = cache ? scope.track(this.backend.concat([cache.v, chunkV], 2)) : chunkV;

    const output = this.attend(q, fullK, fullV, mask);
    const newCache = this.trimCache(fullK, fullV);
    scope.keep(output);
    scope.keep(newCache.k);
    scope.keep(newCache.v);
    scope.dispose(this.backend);

    return {
      output,
      cache: newCache,
    };
  }

  private attend(q: TensorHandle, k: TensorHandle, v: TensorHandle, mask?: TensorHandle): TensorHandle {
    const scope = new ComputeScope();
    const kT = scope.track(this.backend.transpose(k, [0, 1, 3, 2]));
    const logits = scope.track(this.backend.scale(this.backend.matmul(q, kT), 1 / Math.sqrt(this.headDim)));
    const keyLength = this.backend.getShape(k)[2];
    const posScores = scope.track(this.posEncoding.forward(q, keyLength));
    let scores = scope.track(this.backend.add(logits, posScores));

    if (mask) {
      scores = scope.track(this.backend.add(scores, mask));
    }

    const weights = scope.track(this.backend.softmax(scores, -1));
    const context = scope.track(this.backend.matmul(weights, v));
    const merged = scope.track(this.mergeHeads(context));
    const projected = this.outProj.forward(merged);
    scope.keep(projected);
    scope.dispose(this.backend);
    return projected;
  }

  private splitHeads(x: TensorHandle): TensorHandle {
    const [batch, time, modelDim] = this.backend.getShape(x);
    const reshaped = this.backend.reshape(x, [batch, time, this.numHeads, this.headDim]);
    const transposed = this.backend.transpose(reshaped, [0, 2, 1, 3]);
    this.backend.dispose(reshaped);
    if (modelDim !== this.numHeads * this.headDim) {
      throw new Error(
        `Invalid attention dimensions: modelDim=${modelDim}, heads=${this.numHeads}, headDim=${this.headDim}`,
      );
    }
    return transposed;
  }

  private mergeHeads(x: TensorHandle): TensorHandle {
    const [batch, heads, time, headDim] = this.backend.getShape(x);
    const transposed = this.backend.transpose(x, [0, 2, 1, 3]);
    const merged = this.backend.reshape(transposed, [batch, time, heads * headDim]);
    this.backend.dispose(transposed);
    return merged;
  }

  private trimCache(k: TensorHandle, v: TensorHandle): AttentionCache {
    const shape = this.backend.getShape(k);
    const currentLen = shape[2];
    const keepLen = Math.min(currentLen, this.contextLeft);

    if (currentLen === keepLen) {
      return { k, v };
    }

    const start = currentLen - keepLen;
    const kSlice = this.backend.slice(k, [0, 0, start, 0], [shape[0], shape[1], keepLen, shape[3]]);
    const vShape = this.backend.getShape(v);
    const vSlice = this.backend.slice(v, [0, 0, start, 0], [vShape[0], vShape[1], keepLen, vShape[3]]);
    return { k: kSlice, v: vSlice };
  }
}
