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
    let h = b.layerNorm(x, this.w.norm.weight, this.w.norm.bias, 1e-5);

    h = linearForward(b, h, this.w.pointwise1.weight, this.w.pointwise1.bias);
    const [a, g] = b.split(h, 2, -1);
    const gated = b.mul(a, b.sigmoid(g));
    b.dispose(a);
    b.dispose(g);
    b.dispose(h);

    const shape = b.getShape(gated);
    const B = shape[0];
    const T = shape[1];
    const C = shape[2];
    const inp4 = b.expandDims(gated, 1);
    b.dispose(gated);

    const fsh = b.getShape(this.w.depthwiseWeight);
    const Kdim = fsh.length === 3 ? fsh[2] : fsh[1];
    const Cdim = fsh[0];
    const flatW = b.reshape(this.w.depthwiseWeight, [Cdim, Kdim]);
    const tfFilt = b.reshape(flatW, [1, Kdim, Cdim, 1]);
    b.dispose(flatW);
    const dw = b.depthwiseConv2d(inp4, tfFilt, [1, 1], 'same');
    b.dispose(tfFilt);
    b.dispose(inp4);

    let dw3 = b.squeeze(dw, [1]);
    b.dispose(dw);

    if (this.w.depthwiseBias) {
      const biased = b.add(dw3, this.w.depthwiseBias);
      b.dispose(dw3);
      dw3 = biased;
    }

    const nct = b.transpose(dw3, [0, 2, 1]);
    b.dispose(dw3);
    let bn = b.batchNorm(
      nct,
      this.w.batchNorm.mean,
      this.w.batchNorm.variance,
      this.w.batchNorm.scale,
      this.w.batchNorm.offset,
      1e-5,
    );
    b.dispose(nct);
    dw3 = b.transpose(bn, [0, 2, 1]);
    b.dispose(bn);
    dw3 = b.silu(dw3);

    const out = linearForward(b, dw3, this.w.pointwise2.weight, this.w.pointwise2.bias);
    b.dispose(dw3);
    void B;
    void T;
    void C;
    return out;
  }
}
