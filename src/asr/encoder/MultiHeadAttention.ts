import type { ComputeBackend } from '../compute/Backend';
import type { TensorHandle } from '../compute/types';
import type { AttentionWeights } from '../model/WeightMapper';
import { Linear } from './Linear';
import { RelativePositionalEncoding } from './RelativePositionalEncoding';

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
  private posBias: TensorHandle | null;
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
    this.posBias = weights.posBias;
    this.posEncoding = new RelativePositionalEncoding(backend, dModel);
    this.posWeight = weights.posProj ? weights.posProj.weight : null;
  }

  forward(x: TensorHandle, mask?: TensorHandle): TensorHandle {
    return this.backend.tidy(() => {
      const shape = this.backend.getShape(x);
      const B = shape[0] as number;
      const T = shape[1] as number;

      let normed = this.backend.layerNorm(x, this.normWeight, this.normBias, 1e-5);

      let q = this.splitHeads(this.qProj.forward(normed), B, T);
      const k = this.splitHeads(this.kProj.forward(normed), B, T);
      const v = this.splitHeads(this.vProj.forward(normed), B, T);

      // Add position bias to query if available (NeMo uses pos_bias_u and pos_bias_v)
      if (this.posBias) {
        const bias = this.backend.reshape(this.posBias, [1, this.numHeads, 1, this.headDim]);
        q = this.backend.add(q, bias);
      }

      const scale = 1.0 / Math.sqrt(this.headDim);

      // Content-based attention: Q * K^T
      const kT = this.backend.transpose(k, [0, 1, 3, 2]); // [B, heads, d_head, T]
      let scores = this.backend.scale(this.backend.matmul(q, kT), scale);

      // Relative positional encoding scores
      if (this.posWeight) {
        const posEnc = this.posEncoding.forward(T);
        const relScores = this.posEncoding.computeRelativeScores(
          q, posEnc, this.posWeight, this.numHeads, this.headDim
        );
        const scaledRelScores = this.backend.scale(relScores, scale);
        scores = this.backend.add(scores, scaledRelScores);
      }

      // Apply mask if provided
      if (mask) {
        const negInf = this.backend.scalarTensor(-1e9);
        const ones = this.backend.ones(this.backend.getShape(scores));
        const maskedScores = this.backend.mul(ones, negInf);
        scores = this.backend.where(mask, scores, maskedScores);
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

    let q = this.splitHeads(this.qProj.forward(normed), B, T);
    let k = this.splitHeads(this.kProj.forward(normed), B, T);
    let v = this.splitHeads(this.vProj.forward(normed), B, T);

    // Concatenate with cache
    if (cachedK && cachedV) {
      k = this.backend.concat([cachedK, k], 2);
      v = this.backend.concat([cachedV, v], 2);
    }

    const newK = this.backend.clone(k);
    const newV = this.backend.clone(v);

    if (this.posBias) {
      const bias = this.backend.reshape(this.posBias, [1, this.numHeads, 1, this.headDim]);
      q = this.backend.add(q, bias);
    }

    const scale = 1.0 / Math.sqrt(this.headDim);
    const kT = this.backend.transpose(k, [0, 1, 3, 2]);
    let scores = this.backend.scale(this.backend.matmul(q, kT), scale);

    const totalT = this.backend.getShape(k)[2] as number;
    if (this.posWeight) {
      const posEnc = this.posEncoding.forward(totalT);
      const relScores = this.posEncoding.computeRelativeScores(
        q, posEnc, this.posWeight, this.numHeads, this.headDim
      );
      const scaledRelScores = this.backend.scale(relScores, scale);
      // Align relative scores to full K/V length
      const relShape = this.backend.getShape(scaledRelScores);
      const relT = relShape[3] as number;
      if (relT !== totalT) {
        const sliced = this.backend.slice(scaledRelScores, [0, 0, 0, relT - totalT], [B, this.numHeads, T, totalT]);
        scores = this.backend.add(scores, sliced);
      } else {
        scores = this.backend.add(scores, scaledRelScores);
      }
    }

    if (mask) {
      const negInf = this.backend.scalarTensor(-1e9);
      const maskOnes = this.backend.mul(this.backend.ones(this.backend.getShape(scores)), negInf);
      scores = this.backend.where(mask, scores, maskOnes);
    }

    const attnWeights = this.backend.softmax(scores, -1);
    const attnOut = this.backend.matmul(attnWeights, v);
    const merged = this.mergeHeads(attnOut, B, T);
    const output = this.outProj.forward(merged);

    return { output, newK, newV };
  }

  private splitHeads(x: TensorHandle, B: number, T: number): TensorHandle {
    const reshaped = this.backend.reshape(x, [B, T, this.numHeads, this.headDim]);
    return this.backend.transpose(reshaped, [0, 2, 1, 3]); // [B, heads, T, d_head]
  }

  private mergeHeads(x: TensorHandle, B: number, T: number): TensorHandle {
    const transposed = this.backend.transpose(x, [0, 2, 1, 3]); // [B, T, heads, d_head]
    return this.backend.reshape(transposed, [B, T, this.dModel]);
  }
}
