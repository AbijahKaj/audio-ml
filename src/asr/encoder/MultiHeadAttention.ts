import { ComputeScope } from '../compute/Scope';
import type { ComputeBackend } from '../compute/Backend';
import type { TensorHandle } from '../compute/types';
import type { AttentionWeights } from '../model/WeightMapper';
import { Linear } from './Linear';
import { RelativePositionalEncoding } from './RelativePositionalEncoding';

export class MultiHeadAttention {
  private readonly qProj: Linear;
  private readonly kProj: Linear;
  private readonly vProj: Linear;
  private readonly outProj: Linear;
  private readonly posEncoding: RelativePositionalEncoding;
  private readonly numHeads: number;

  constructor(
    private readonly backend: ComputeBackend,
    private readonly weights: AttentionWeights,
    numHeads: number,
  ) {
    this.qProj = new Linear(backend, weights.qProj);
    this.kProj = new Linear(backend, weights.kProj);
    this.vProj = new Linear(backend, weights.vProj);
    this.outProj = new Linear(backend, weights.outProj);
    this.posEncoding = new RelativePositionalEncoding(backend);
    this.numHeads = Math.max(1, numHeads);
  }

  forward(input: TensorHandle, _mask?: TensorHandle): TensorHandle {
    if (!this.weights.qProj.weight || !this.weights.kProj.weight || !this.weights.vProj.weight) {
      return input;
    }

    const scope = new ComputeScope();
    const normalized = this.weights.norm.weight && this.weights.norm.bias
      ? scope.track(this.backend.layerNorm(input, this.weights.norm.weight, this.weights.norm.bias, 1e-5))
      : input;
    const q = scope.track(this.splitHeads(this.qProj.forward(normalized)));
    const k = scope.track(this.splitHeads(this.kProj.forward(normalized)));
    const v = scope.track(this.splitHeads(this.vProj.forward(normalized)));
    const kT = scope.track(this.backend.transpose(k, [0, 1, 3, 2]));
    const scores = scope.track(this.backend.scale(this.backend.matmul(q, kT), 1 / Math.sqrt(this.headDim(input))));
    const pos = scope.track(this.posEncoding.forward(q, this.backend.getShape(input)[1]));
    const biasedScores = scope.track(this.backend.add(scores, pos));
    const weights = scope.track(this.backend.softmax(biasedScores, -1));
    const attended = scope.track(this.backend.matmul(weights, v));
    const merged = scope.track(this.mergeHeads(attended));
    const output = this.outProj.forward(merged);
    scope.dispose(this.backend);
    return output;
  }

  private headDim(input: TensorHandle): number {
    const dModel = this.backend.getShape(input)[2];
    return dModel % this.numHeads === 0 ? dModel / this.numHeads : dModel;
  }

  private splitHeads(x: TensorHandle): TensorHandle {
    const [batch, time, dModel] = this.backend.getShape(x);
    const headDim = dModel % this.numHeads === 0 ? dModel / this.numHeads : dModel;
    const heads = dModel % this.numHeads === 0 ? this.numHeads : 1;
    const reshaped = this.backend.reshape(x, [batch, time, heads, headDim]);
    const transposed = this.backend.transpose(reshaped, [0, 2, 1, 3]);
    this.backend.dispose(reshaped);
    return transposed;
  }

  private mergeHeads(x: TensorHandle): TensorHandle {
    const [batch, heads, time, headDim] = this.backend.getShape(x);
    const transposed = this.backend.transpose(x, [0, 2, 1, 3]);
    const reshaped = this.backend.reshape(transposed, [batch, time, heads * headDim]);
    this.backend.dispose(transposed);
    return reshaped;
  }
}
