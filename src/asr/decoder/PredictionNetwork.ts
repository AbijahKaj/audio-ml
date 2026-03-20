import type { ComputeBackend } from '../compute/Backend';
import type { TensorHandle } from '../compute/types';
import type { PredictionWeights } from '../model/types';

/**
 * Single-layer LSTM prediction network (NeMo RNNT decoder).
 */
export class PredictionNetwork {
  private readonly hiddenSize: number;
  private readonly embedDim: number;

  constructor(
    private readonly backend: ComputeBackend,
    private readonly w: PredictionWeights,
  ) {
    const es = this.backend.getShape(w.embedding);
    this.embedDim = es[1]!;
    const hs = this.backend.getShape(w.lstmWhh);
    this.hiddenSize = hs[1]!;
  }

  initialState(): { h: TensorHandle; c: TensorHandle } {
    return {
      h: this.backend.zeros([1, this.hiddenSize]),
      c: this.backend.zeros([1, this.hiddenSize]),
    };
  }

  /**
   * One LSTM step. `tokenId` selects an embedding row; `h`/`c` are [1, H].
   */
  step(
    tokenId: number,
    h: TensorHandle,
    c: TensorHandle,
  ): { h: TensorHandle; c: TensorHandle } {
    const idx = this.backend.tensor(new Int32Array([tokenId]), [1], 'int32');
    const emb = this.backend.gather(this.w.embedding, idx, 0);
    this.backend.dispose(idx);
    const emb2 = this.backend.reshape(emb, [1, this.embedDim]);
    this.backend.dispose(emb);

    const wihT = this.backend.transpose(this.w.lstmWih, [1, 0]);
    const whhT = this.backend.transpose(this.w.lstmWhh, [1, 0]);
    const g1 = this.backend.matmul(emb2, wihT);
    const g2 = this.backend.matmul(h, whhT);
    this.backend.dispose(wihT);
    this.backend.dispose(whhT);
    const gates = this.backend.add(this.backend.add(g1, g2), this.w.lstmBias);
    this.backend.dispose(g1);
    this.backend.dispose(g2);
    this.backend.dispose(emb2);

    const H = this.hiddenSize;
    const gi = this.backend.slice(gates, [0, 0], [1, H]);
    const gf = this.backend.slice(gates, [0, H], [1, H]);
    const gg = this.backend.slice(gates, [0, 2 * H], [1, H]);
    const go = this.backend.slice(gates, [0, 3 * H], [1, H]);
    this.backend.dispose(gates);

    const i = this.backend.sigmoid(gi);
    const f = this.backend.sigmoid(gf);
    const g = this.backend.tanh(gg);
    const o = this.backend.sigmoid(go);
    this.backend.dispose(gi);
    this.backend.dispose(gf);
    this.backend.dispose(gg);
    this.backend.dispose(go);

    const cNew = this.backend.add(this.backend.mul(f, c), this.backend.mul(i, g));
    this.backend.dispose(f);
    this.backend.dispose(i);
    this.backend.dispose(g);
    const tanhC = this.backend.tanh(cNew);
    const hNew = this.backend.mul(o, tanhC);
    this.backend.dispose(o);
    this.backend.dispose(tanhC);
    this.backend.dispose(h);
    this.backend.dispose(c);

    return { h: hNew, c: cNew };
  }
}