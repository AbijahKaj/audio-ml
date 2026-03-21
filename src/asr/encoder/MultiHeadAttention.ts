import type { ComputeBackend } from '../compute/Backend';
import type { TensorHandle } from '../compute/types';
import type { AttentionWeights } from '../model/WeightMapper';
import { Linear } from './Linear';
import { RelativePositionalEncoding } from './RelativePositionalEncoding';

/**
 * Multi-head self-attention with relative positional encoding.
 * Uses two position biases (pos_bias_u for content and pos_bias_v for position)
 * per the original Conformer paper.
 */
export class MultiHeadAttention {
  private backend: ComputeBackend;
  private numHeads: number;
  private headDim: number;
  private dModel: number;
  private normWeight: TensorHandle;
  private normBias: TensorHandle;
  private qProj: Linear;
  private kProj: Linear;
  private vProj: Linear;
  private outProj: Linear;
  private posBiasU: TensorHandle;
  private posBiasV: TensorHandle;
  private posEncoding: RelativePositionalEncoding;
  private posWeight: TensorHandle | null;

  constructor(
    backend: ComputeBackend,
    weights: AttentionWeights,
    numHeads: number,
    dModel: number,
  ) {
    this.backend = backend;
    this.numHeads = numHeads;
    this.dModel = dModel;
    this.headDim = dModel / numHeads;
    this.normWeight = weights.norm.weight;
    this.normBias = weights.norm.bias;
    this.qProj = new Linear(backend, weights.queryProj);
    this.kProj = new Linear(backend, weights.keyProj);
    this.vProj = new Linear(backend, weights.valueProj);
    this.outProj = new Linear(backend, weights.outProj);
    this.posBiasU = weights.posBiasU;
    this.posBiasV = weights.posBiasV;
    this.posEncoding = new RelativePositionalEncoding(backend, dModel);
    this.posWeight = weights.posProj ? weights.posProj.weight : null;
  }

  forward(x: TensorHandle, mask?: TensorHandle): TensorHandle {
    return this.backend.tidy(() => {
      const shape = this.backend.getShape(x);
      const B = shape[0] as number;
      const T = shape[1] as number;

      const normed = this.backend.layerNorm(x, this.normWeight, this.normBias, 1e-5);

      const q = this.splitHeads(this.qProj.forward(normed), B, T);
      const k = this.splitHeads(this.kProj.forward(normed), B, T);
      const v = this.splitHeads(this.vProj.forward(normed), B, T);

      const scale = 1.0 / Math.sqrt(this.headDim);

      // pos_bias_u: [heads, d_head] -> [1, heads, 1, d_head]
      const biasU = this.backend.reshape(this.posBiasU, [1, this.numHeads, 1, this.headDim]);
      const biasV = this.backend.reshape(this.posBiasV, [1, this.numHeads, 1, this.headDim]);

      // Content-based attention: (Q + pos_bias_u) * K^T
      const qWithBiasU = this.backend.add(q, biasU);
      const kT = this.backend.transpose(k, [0, 1, 3, 2]);
      let contentScores = this.backend.matmul(qWithBiasU, kT);

      // Position-based attention: (Q + pos_bias_v) * pos_encoding^T
      if (this.posWeight && T > 0) {
        const qWithBiasV = this.backend.add(q, biasV);
        const posEnc = this.posEncoding.forward(T);
        const relScores = this.posEncoding.computeRelativeScores(
          qWithBiasV, posEnc, this.posWeight, this.numHeads, this.headDim
        );
        contentScores = this.backend.add(contentScores, relScores);
      }

      let scores = this.backend.scale(contentScores, scale);

      if (mask) {
        const negInf = this.backend.scalarTensor(-1e9);
        const maskFill = this.backend.mul(this.backend.ones(this.backend.getShape(scores)), negInf);
        scores = this.backend.where(mask, scores, maskFill);
      }

      const attnWeights = this.backend.softmax(scores, -1);
      const attnOut = this.backend.matmul(attnWeights, v);

      const merged = this.mergeHeads(attnOut, B, T);
      return this.outProj.forward(merged);
    });
  }

  forwardStreaming(
    x: TensorHandle,
    cachedK: TensorHandle | null,
    cachedV: TensorHandle | null,
    mask?: TensorHandle,
  ): { output: TensorHandle; newK: TensorHandle; newV: TensorHandle } {
    const shape = this.backend.getShape(x);
    const B = shape[0] as number;
    const T = shape[1] as number;

    const normed = this.backend.layerNorm(x, this.normWeight, this.normBias, 1e-5);

    const q = this.splitHeads(this.qProj.forward(normed), B, T);
    let k = this.splitHeads(this.kProj.forward(normed), B, T);
    let v = this.splitHeads(this.vProj.forward(normed), B, T);

    if (cachedK && cachedV) {
      k = this.backend.concat([cachedK, k], 2);
      v = this.backend.concat([cachedV, v], 2);
    }

    const newK = this.backend.clone(k);
    const newV = this.backend.clone(v);

    const scale = 1.0 / Math.sqrt(this.headDim);
    const biasU = this.backend.reshape(this.posBiasU, [1, this.numHeads, 1, this.headDim]);
    const qWithBiasU = this.backend.add(q, biasU);
    const kT = this.backend.transpose(k, [0, 1, 3, 2]);
    let scores = this.backend.matmul(qWithBiasU, kT);

    const totalT = this.backend.getShape(k)[2] as number;
    if (this.posWeight && totalT > 0) {
      const biasV = this.backend.reshape(this.posBiasV, [1, this.numHeads, 1, this.headDim]);

      if (T === totalT) {
        const qWithBiasV = this.backend.add(q, biasV);
        const posEnc = this.posEncoding.forward(totalT);
        const relScores = this.posEncoding.computeRelativeScores(
          qWithBiasV, posEnc, this.posWeight, this.numHeads, this.headDim
        );
        scores = this.backend.add(scores, relScores);
      } else {
        const qPadded = this.backend.pad(q, [[0, 0], [0, 0], [totalT - T, 0], [0, 0]]);
        const qPaddedBiased = this.backend.add(qPadded, biasV);
        const posEnc = this.posEncoding.forward(totalT);
        const fullRelScores = this.posEncoding.computeRelativeScores(
          qPaddedBiased, posEnc, this.posWeight, this.numHeads, this.headDim
        );
        const relSliced = this.backend.slice(
          fullRelScores, [0, 0, totalT - T, 0], [B, this.numHeads, T, totalT]
        );
        scores = this.backend.add(scores, relSliced);
      }
    }

    scores = this.backend.scale(scores, scale);

    if (mask) {
      const negInf = this.backend.scalarTensor(-1e9);
      const maskFill = this.backend.mul(this.backend.ones(this.backend.getShape(scores)), negInf);
      scores = this.backend.where(mask, scores, maskFill);
    }

    const attnWeights = this.backend.softmax(scores, -1);
    const attnOut = this.backend.matmul(attnWeights, v);
    const merged = this.mergeHeads(attnOut, B, T);
    const output = this.outProj.forward(merged);

    return { output, newK, newV };
  }

  private splitHeads(x: TensorHandle, B: number, T: number): TensorHandle {
    const reshaped = this.backend.reshape(x, [B, T, this.numHeads, this.headDim]);
    return this.backend.transpose(reshaped, [0, 2, 1, 3]);
  }

  private mergeHeads(x: TensorHandle, B: number, T: number): TensorHandle {
    const transposed = this.backend.transpose(x, [0, 2, 1, 3]);
    return this.backend.reshape(transposed, [B, T, this.dModel]);
  }
}
