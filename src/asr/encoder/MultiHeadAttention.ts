import type { ComputeBackend } from '../compute/Backend';
import { ComputeScope } from '../compute/ComputeScope';
import type { FastConformerConfig } from '../model/ModelConfig';
import type { SelfAttentionWeights } from '../model/weights';
import type { TensorHandle } from '../compute/types';
import { ATT_MASK } from './constants';
import { linearForward } from './ops';

export class MultiHeadAttention {
  private readonly dHead: number;
  private readonly scale: number;

  constructor(
    private readonly backend: ComputeBackend,
    private readonly w: SelfAttentionWeights,
    private readonly config: FastConformerConfig,
  ) {
    this.dHead = config.dModel / config.numHeads;
    this.scale = 1 / Math.sqrt(this.dHead);
  }

  forward(x: TensorHandle, posEmb: TensorHandle, padMask?: TensorHandle): TensorHandle {
    if (this.config.selfAttentionModel === 'rel_pos_local_attn') {
      return this.forwardLocal(x, posEmb, padMask);
    }
    if (this.config.selfAttentionModel === 'rel_pos') {
      return this.forwardFullRelPos(x, posEmb, padMask);
    }
    return this.forwardAbsPos(x, padMask);
  }

  private qkv(x: TensorHandle): [TensorHandle, TensorHandle, TensorHandle] {
    const b = this.backend;
    const scope = new ComputeScope();
    const q = scope.track(linearForward(b, x, this.w.linearQ.weight, this.w.linearQ.bias));
    const k = scope.track(linearForward(b, x, this.w.linearK.weight, this.w.linearK.bias));
    const v = scope.track(linearForward(b, x, this.w.linearV.weight, this.w.linearV.bias));
    scope.dispose(b);
    const B = b.getShape(x)[0];
    const T = b.getShape(x)[1];
    const H = this.config.numHeads;
    const D = this.dHead;
    const qh = b.reshape(q, [B, T, H, D]);
    const kh = b.reshape(k, [B, T, H, D]);
    const vh = b.reshape(v, [B, T, H, D]);
    const qt = b.transpose(qh, [0, 2, 1, 3]);
    const kt = b.transpose(kh, [0, 2, 1, 3]);
    const vt = b.transpose(vh, [0, 2, 1, 3]);
    b.dispose(q);
    b.dispose(k);
    b.dispose(v);
    b.dispose(qh);
    b.dispose(kh);
    b.dispose(vh);
    return [qt, kt, vt];
  }

  /** Local context attention (FastConformer streaming / Parakeet). */
  private forwardLocal(x: TensorHandle, posEmb: TensorHandle, padMask?: TensorHandle): TensorHandle {
    const b = this.backend;
    const [left, right] = this.config.attContextSize;
    const nOff = left + right + 1;
    const [q, k, v] = this.qkv(x);
    const shape = b.getShape(q);
    const B = shape[0];
    const H = shape[1];
    const T = shape[2];
    const D = shape[3];

    const posBiasU = b.reshape(this.w.posBiasU, [1, H, 1, D]);
    const posBiasV = b.reshape(this.w.posBiasV, [1, H, 1, D]);
    const qU = b.add(q, posBiasU);
    const qV = b.add(q, posBiasV);

    const kPad = b.pad(k, [
      [0, 0],
      [0, 0],
      [left, right],
      [0, 0],
    ]);
    const vPad = b.pad(v, [
      [0, 0],
      [0, 0],
      [left, right],
      [0, 0],
    ]);

    const kSlices: TensorHandle[] = [];
    const vSlices: TensorHandle[] = [];
    for (let o = 0; o < nOff; o++) {
      kSlices.push(b.slice(kPad, [0, 0, o, 0], [B, H, T, D]));
      vSlices.push(b.slice(vPad, [0, 0, o, 0], [B, H, T, D]));
    }
    const kStack = b.stack(kSlices, 3);
    const vStack = b.stack(vSlices, 3);
    for (const t of kSlices) b.dispose(t);
    for (const t of vSlices) b.dispose(t);
    b.dispose(kPad);
    b.dispose(vPad);

    const qe = b.expandDims(qU, 3);
    const acScores = b.scale(b.sum(b.mul(qe, kStack), -1), this.scale);
    b.dispose(qe);
    b.dispose(kStack);

    const pLin = linearForward(b, posEmb, this.w.linearPos.weight, this.w.linearPos.bias);
    const pShape = b.getShape(pLin);
    const pHeads = b.reshape(pLin, [pShape[0], pShape[1], H, D]);
    const pBhtd = b.transpose(pHeads, [0, 2, 1, 3]);
    b.dispose(pLin);
    b.dispose(pHeads);
    const pTd = b.transpose(pBhtd, [0, 1, 3, 2]);
    const bd = b.scale(b.matmul(qV, pTd), this.scale);
    b.dispose(pBhtd);
    b.dispose(pTd);

    let scores = b.add(acScores, bd);
    b.dispose(acScores);
    b.dispose(bd);

    if (padMask) {
      const m = b.expandDims(b.expandDims(padMask, 1), 3);
      const maskVal = b.mul(b.cast(m, 'float32'), ATT_MASK);
      scores = b.add(scores, maskVal);
      b.dispose(m);
      b.dispose(maskVal);
    }

    const attn = b.softmax(scores, -1);
    b.dispose(scores);

    const av = b.expandDims(attn, -1);
    const out = b.sum(b.mul(av, vStack), 3);
    b.dispose(av);
    b.dispose(vStack);
    b.dispose(attn);

    const outBt = b.transpose(out, [0, 2, 1, 3]);
    b.dispose(out);
    const merged = b.reshape(outBt, [B, T, this.config.dModel]);
    b.dispose(outBt);
    const proj = linearForward(b, merged, this.w.linearOut.weight, this.w.linearOut.bias);
    b.dispose(merged);
    b.dispose(q);
    b.dispose(k);
    b.dispose(v);
    b.dispose(qU);
    b.dispose(qV);
    b.dispose(posBiasU);
    b.dispose(posBiasV);
    return proj;
  }

  private forwardAbsPos(x: TensorHandle, padMask?: TensorHandle): TensorHandle {
    const b = this.backend;
    const [q, k, v] = this.qkv(x);
    const scores = b.scale(b.matmul(q, b.transpose(k, [0, 1, 3, 2])), this.scale);
    let s = scores;
    if (padMask) {
      const m = b.expandDims(padMask, 1);
      const add = b.mul(b.cast(m, 'float32'), ATT_MASK);
      s = b.add(scores, add);
      b.dispose(add);
      b.dispose(m);
    }
    const attn = b.softmax(s, -1);
    b.dispose(s);
    const ctx = b.matmul(attn, v);
    b.dispose(attn);
    const shape = b.getShape(x);
    const B = shape[0];
    const T = shape[1];
    const merged = b.reshape(b.transpose(ctx, [0, 2, 1, 3]), [B, T, this.config.dModel]);
    b.dispose(ctx);
    b.dispose(q);
    b.dispose(k);
    b.dispose(v);
    const out = linearForward(b, merged, this.w.linearOut.weight, this.w.linearOut.bias);
    b.dispose(merged);
    return out;
  }

  /** Full relative positional (Transformer-XL style) — offline. */
  private forwardFullRelPos(x: TensorHandle, posEmb: TensorHandle, padMask?: TensorHandle): TensorHandle {
    const b = this.backend;
    const [q, k, v] = this.qkv(x);
    const shape = b.getShape(q);
    const B = shape[0];
    const H = shape[1];
    const T = shape[2];
    const D = this.dHead;

    const posBiasU = b.reshape(this.w.posBiasU, [1, H, 1, D]);
    const posBiasV = b.reshape(this.w.posBiasV, [1, H, 1, D]);
    const qU = b.add(q, posBiasU);
    const qV = b.add(q, posBiasV);

    const ac = b.scale(b.matmul(qU, b.transpose(k, [0, 1, 3, 2])), this.scale);

    const pLin = linearForward(b, posEmb, this.w.linearPos.weight, this.w.linearPos.bias);
    const peShape = b.getShape(pLin);
    const pH = b.reshape(pLin, [peShape[0], peShape[1], H, D]);
    const pT = b.transpose(pH, [0, 2, 3, 1]);
    b.dispose(pLin);
    b.dispose(pH);
    let bd = b.matmul(qV, pT);
    bd = b.scale(bd, this.scale);
    bd = this.relShift(bd, B, H, T);
    b.dispose(pT);

    const acShape = b.getShape(ac);
    const bdSliced = b.slice(bd, [0, 0, 0, 0], [B, H, T, acShape[3]]);
    b.dispose(bd);
    let scores = b.add(ac, bdSliced);
    b.dispose(ac);
    b.dispose(bdSliced);

    if (padMask) {
      const m = b.expandDims(padMask, 1);
      const add = b.mul(b.cast(m, 'float32'), ATT_MASK);
      scores = b.add(scores, add);
      b.dispose(add);
      b.dispose(m);
    }

    const attn = b.softmax(scores, -1);
    b.dispose(scores);
    const ctx = b.matmul(attn, v);
    b.dispose(attn);
    const merged = b.reshape(b.transpose(ctx, [0, 2, 1, 3]), [B, T, this.config.dModel]);
    b.dispose(ctx);
    b.dispose(q);
    b.dispose(k);
    b.dispose(v);
    b.dispose(qU);
    b.dispose(qV);
    b.dispose(posBiasU);
    b.dispose(posBiasV);
    const out = linearForward(b, merged, this.w.linearOut.weight, this.w.linearOut.bias);
    b.dispose(merged);
    return out;
  }

  private relShift(x: TensorHandle, B: number, H: number, qlen: number): TensorHandle {
    const b = this.backend;
    const posLen = b.getShape(x)[3];
    const padded = b.pad(x, [
      [0, 0],
      [0, 0],
      [0, 0],
      [1, 0],
    ]);
    const resh = b.reshape(padded, [B, H, -1, qlen]);
    const sliced = b.slice(resh, [0, 0, 1, 0], [B, H, b.getShape(resh)[2] - 1, qlen]);
    b.dispose(padded);
    b.dispose(resh);
    const out = b.reshape(sliced, [B, H, qlen, posLen]);
    b.dispose(sliced);
    return out;
  }
}
