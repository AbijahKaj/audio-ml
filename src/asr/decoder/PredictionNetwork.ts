import type { ComputeBackend } from '../compute/Backend';
import type { PredictionWeights } from '../model/weights';
import type { TensorHandle } from '../compute/types';
import { embeddingRow, linearForward } from '../encoder/ops';

export class PredictionNetwork {
  constructor(
    private readonly backend: ComputeBackend,
    private readonly w: PredictionWeights,
    private readonly hiddenSize: number,
  ) {}

  initialState(): { h: TensorHandle; c: TensorHandle } {
    const b = this.backend;
    return {
      h: b.zeros([1, 1, this.hiddenSize]),
      c: b.zeros([1, 1, this.hiddenSize]),
    };
  }

  step(
    tokenId: number,
    h: TensorHandle,
    c: TensorHandle,
  ): { output: TensorHandle; h: TensorHandle; c: TensorHandle } {
    const b = this.backend;
    const emb = embeddingRow(b, this.w.embedding, tokenId);
    const g1 = linearForward(b, emb, this.w.lstm.weightIh, this.w.lstm.biasIh);
    const g2 = linearForward(b, h, this.w.lstm.weightHh, this.w.lstm.biasHh);
    b.dispose(emb);
    const gates = b.add(g1, g2);
    b.dispose(g1);
    b.dispose(g2);

    const parts = b.split(gates, 4, 2);
    b.dispose(gates);
    const i = parts[0];
    const f = parts[1];
    const g = parts[2];
    const o = parts[3];

    const iAct = b.sigmoid(i);
    const fAct = b.sigmoid(f);
    const gAct = b.tanh(g);
    const oAct = b.sigmoid(o);
    b.dispose(i);
    b.dispose(f);
    b.dispose(g);
    b.dispose(o);

    const cNew = b.add(b.mul(fAct, c), b.mul(iAct, gAct));
    b.dispose(fAct);
    b.dispose(iAct);
    b.dispose(gAct);
    const hNew = b.mul(oAct, b.tanh(cNew));
    b.dispose(oAct);

    return { output: hNew, h: hNew, c: cNew };
  }
}
