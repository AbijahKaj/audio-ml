import type { ComputeBackend } from '../compute/Backend';
import type { TensorHandle } from '../compute/types';
import type { ConformerConvWeights, LayerNormWeights } from '../model/types';
import { pytorchConv1dToTfjs, pytorchDepthwiseConv1dToTfjs } from './pytorchLayout';

/**
 * Conformer convolution module (GLU → depthwise conv → BN → Swish → pointwise), NWC layout [B, T, C].
 */
export class ConvModule {
  constructor(
    private readonly backend: ComputeBackend,
    private readonly norm: LayerNormWeights,
    private readonly weights: ConformerConvWeights,
    private readonly eps = 1e-5,
  ) {}

  forward(x: TensorHandle): TensorHandle {
    const n = this.backend.layerNorm(x, this.norm.weight, this.norm.bias, this.eps);

    const pw1 = pytorchConv1dToTfjs(this.backend, this.weights.pointwise1.weight);
    const h1 = this.backend.conv1d(n, pw1, 1, 'valid');
    this.backend.dispose(pw1);
    const h1b = this.backend.add(
      h1,
      this.backend.reshape(this.weights.pointwise1.bias, [1, 1, -1]),
    );
    this.backend.dispose(h1);

    const glu = this.glu(h1b);
    this.backend.dispose(h1b);

    const dw = pytorchDepthwiseConv1dToTfjs(this.backend, this.weights.depthwise);
    const h2 = this.backend.depthwiseConv1d(glu, dw, 1, 'same');
    this.backend.dispose(dw);
    const h2b = this.backend.add(h2, this.backend.reshape(this.weights.depthwiseBias, [1, 1, -1]));
    this.backend.dispose(h2);

    const bn = this.backend.batchNorm(
      h2b,
      this.weights.batchNorm.runningMean,
      this.weights.batchNorm.runningVar,
      this.weights.batchNorm.weight,
      this.weights.batchNorm.bias,
      this.eps,
    );
    this.backend.dispose(h2b);

    const sw = this.backend.silu(bn);
    this.backend.dispose(bn);

    const pw2 = pytorchConv1dToTfjs(this.backend, this.weights.pointwise2.weight);
    const h3 = this.backend.conv1d(sw, pw2, 1, 'valid');
    this.backend.dispose(sw);
    this.backend.dispose(pw2);
    const out = this.backend.add(h3, this.backend.reshape(this.weights.pointwise2.bias, [1, 1, -1]));
    this.backend.dispose(h3);
    return out;
  }

  private glu(x: TensorHandle): TensorHandle {
    const s = this.backend.getShape(x);
    const c = s[2]!;
    const half = c / 2;
    const a = this.backend.slice(x, [0, 0, 0], [s[0]!, s[1]!, half]);
    const b = this.backend.slice(x, [0, 0, half], [s[0]!, s[1]!, half]);
    const sig = this.backend.sigmoid(b);
    this.backend.dispose(b);
    const out = this.backend.mul(a, sig);
    this.backend.dispose(a);
    this.backend.dispose(sig);
    return out;
  }
}