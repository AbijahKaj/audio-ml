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
    const temps: TensorHandle[] = [];

    const shape = this.backend.getShape(x);
    const B = shape[0] as number;
    const T = shape[1] as number;

    const normed = this.backend.layerNorm(x, this.normWeight, this.normBias, 1e-5);
    temps.push(normed);

    const qRaw = this.qProj.forward(normed);
    temps.push(qRaw);
    const q = this.splitHeads(qRaw, B, T);
    temps.push(q);

    const kRaw = this.kProj.forward(normed);
    temps.push(kRaw);
    let kHeads = this.splitHeads(kRaw, B, T);
    temps.push(kHeads);

    const vRaw = this.vProj.forward(normed);
    temps.push(vRaw);
    let vHeads = this.splitHeads(vRaw, B, T);
    temps.push(vHeads);

    if (cachedK && cachedV) {
      const kConcat = this.backend.concat([cachedK, kHeads], 2);
      const vConcat = this.backend.concat([cachedV, vHeads], 2);
      temps.push(kConcat, vConcat);
      kHeads = kConcat;
      vHeads = vConcat;
    }

    const newK = this.backend.clone(kHeads);
    const newV = this.backend.clone(vHeads);

    const scale = 1.0 / Math.sqrt(this.headDim);
    const biasU = this.backend.reshape(this.posBiasU, [1, this.numHeads, 1, this.headDim]);
    temps.push(biasU);
    const qWithBiasU = this.backend.add(q, biasU);
    temps.push(qWithBiasU);
    const kT = this.backend.transpose(kHeads, [0, 1, 3, 2]);
    temps.push(kT);
    let scores = this.backend.matmul(qWithBiasU, kT);
    temps.push(scores);

    const totalT = this.backend.getShape(kHeads)[2] as number;
    if (this.posWeight && totalT > 0) {
      const biasV = this.backend.reshape(this.posBiasV, [1, this.numHeads, 1, this.headDim]);
      temps.push(biasV);

      if (T === totalT) {
        const qWithBiasV = this.backend.add(q, biasV);
        temps.push(qWithBiasV);
        const posEnc = this.posEncoding.forward(totalT);
        temps.push(posEnc);
        const relScores = this.posEncoding.computeRelativeScores(
          qWithBiasV, posEnc, this.posWeight, this.numHeads, this.headDim
        );
        temps.push(relScores);
        scores = this.backend.add(scores, relScores);
        temps.push(scores);
      } else {
        const qPadded = this.backend.pad(q, [[0, 0], [0, 0], [totalT - T, 0], [0, 0]]);
        temps.push(qPadded);
        const qPaddedBiased = this.backend.add(qPadded, biasV);
        temps.push(qPaddedBiased);
        const posEnc = this.posEncoding.forward(totalT);
        temps.push(posEnc);
        const fullRelScores = this.posEncoding.computeRelativeScores(
          qPaddedBiased, posEnc, this.posWeight, this.numHeads, this.headDim
        );
        temps.push(fullRelScores);
        const relSliced = this.backend.slice(
          fullRelScores, [0, 0, totalT - T, 0], [B, this.numHeads, T, totalT]
        );
        temps.push(relSliced);
        scores = this.backend.add(scores, relSliced);
        temps.push(scores);
      }
    }

    const scaledScores = this.backend.scale(scores, scale);
    temps.push(scaledScores);

    let finalScores = scaledScores;
    if (mask) {
      const negInf = this.backend.scalarTensor(-1e9);
      temps.push(negInf);
      const onesT = this.backend.ones(this.backend.getShape(scaledScores));
      temps.push(onesT);
      const maskFill = this.backend.mul(onesT, negInf);
      temps.push(maskFill);
      finalScores = this.backend.where(mask, scaledScores, maskFill);
      temps.push(finalScores);
    }

    const attnWeights = this.backend.softmax(finalScores, -1);
    temps.push(attnWeights);
    const attnOut = this.backend.matmul(attnWeights, vHeads);
    temps.push(attnOut);
    const merged = this.mergeHeads(attnOut, B, T);
    temps.push(merged);
    const output = this.outProj.forward(merged);

    for (const t of temps) this.backend.dispose(t);

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
