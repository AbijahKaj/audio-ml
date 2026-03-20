import type { ComputeBackend } from '../compute/Backend';
import type { ConvModuleWeights } from '../model/weights';
import type { TensorHandle } from '../compute/types';
import { linearForward } from './ops';

export class ConvModule {
  constructor(
    private readonly backend: ComputeBackend,
    private readonly w: ConvModuleWeights,
  ) {}

  forward(x: TensorHandle): TensorHandle {
    const b = this.backend;
    const sh1 = b.getShape(this.w.pointwise1.weight);
    const pw1W =
      sh1.length === 3 ? b.reshape(this.w.pointwise1.weight, [sh1[0]!, sh1[1]!]) : this.w.pointwise1.weight;
    let h = linearForward(b, x, pw1W, this.w.pointwise1.bias);
    if (pw1W !== this.w.pointwise1.weight) {
      b.dispose(pw1W);
    }

    const [a, g] = b.split(h, 2, -1);
    const gated = b.mul(a, b.sigmoid(g));
    b.dispose(a);
    b.dispose(g);
    b.dispose(h);

    const shape = b.getShape(gated);
    const B = shape[0]!;
    const T = shape[1]!;
    const D = shape[2]!;
    const inp4 = b.reshape(gated, [B, T, 1, D]);
    b.dispose(gated);

    const fsh = b.getShape(this.w.depthwiseWeight);
    const k = fsh[2]!;
    const cOut = fsh[0]!;
    const flatW = b.reshape(this.w.depthwiseWeight, [cOut, k]);
    const tfF = b.reshape(flatW, [k, 1, cOut, 1]);
    b.dispose(flatW);
    let dw3 = b.depthwiseConv2d(inp4, tfF, [1, 1], 'same');
    b.dispose(tfF);
    b.dispose(inp4);

    if (this.w.depthwiseBias) {
      const biased = b.add(dw3, this.w.depthwiseBias);
      b.dispose(dw3);
      dw3 = biased;
    }

    const normBias = this.w.afterDepthwise.bias ?? b.zeros(b.getShape(this.w.afterDepthwise.weight));
    if (this.w.useBatchNormStats && this.w.bnRunningMean && this.w.bnRunningVar) {
      const nct = b.transpose(dw3, [0, 2, 1]);
      b.dispose(dw3);
      let bn = b.batchNorm(
        nct,
        this.w.bnRunningMean,
        this.w.bnRunningVar,
        this.w.afterDepthwise.weight,
        normBias,
        1e-5,
      );
      b.dispose(nct);
      dw3 = b.transpose(bn, [0, 2, 1]);
      b.dispose(bn);
    } else {
      dw3 = b.layerNorm(dw3, this.w.afterDepthwise.weight, normBias, 1e-5);
    }
    if (!this.w.afterDepthwise.bias) {
      b.dispose(normBias);
    }

    dw3 = b.silu(dw3);

    const sh2 = b.getShape(this.w.pointwise2.weight);
    const pw2W =
      sh2.length === 3 ? b.reshape(this.w.pointwise2.weight, [sh2[0]!, sh2[1]!]) : this.w.pointwise2.weight;
    const out = linearForward(b, dw3, pw2W, this.w.pointwise2.bias);
    if (pw2W !== this.w.pointwise2.weight) {
      b.dispose(pw2W);
    }
    b.dispose(dw3);
    void B;
    void T;
    void D;
    return out;
  }
}
