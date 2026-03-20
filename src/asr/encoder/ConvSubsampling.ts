import type { ComputeBackend } from '../compute/Backend';
import type { TensorHandle } from '../compute/types';
import type { LinearWeights } from '../model/types';
import { linearForward } from './linear';
import { pytorchConv2dToTfjs } from './pytorchLayout';

export interface Conv2dBiasWeights {
  weight: TensorHandle;
  bias: TensorHandle;
}

/**
 * NeMo-style striding Conv2d subsampling (3× stride-2 Conv2d + linear), input [B, T, F].
 */
export class ConvSubsampling {
  constructor(
    private readonly backend: ComputeBackend,
    private readonly conv0: Conv2dBiasWeights,
    private readonly conv1: Conv2dBiasWeights,
    private readonly conv2: Conv2dBiasWeights,
    private readonly outProj: LinearWeights,
  ) {}

  forward(x: TensorHandle): TensorHandle {
    const s0 = this.backend.getShape(x);
    const b = s0[0]!;
    const t = s0[1]!;
    const f = s0[2]!;

    let h = this.backend.reshape(x, [b, t, f, 1]);

    h = this.conv2dRelu(h, this.conv0);
    h = this.conv2dRelu(h, this.conv1);
    h = this.conv2dRelu(h, this.conv2);

    const sh = this.backend.getShape(h);
    const t2 = sh[1]!;
    const f2 = sh[2]!;
    const c = sh[3]!;

    const flat = this.backend.reshape(h, [b, t2, c * f2]);
    this.backend.dispose(h);

    const out = linearForward(this.backend, flat, this.outProj);
    this.backend.dispose(flat);
    return out;
  }

  private conv2dRelu(x: TensorHandle, w: Conv2dBiasWeights): TensorHandle {
    const k = pytorchConv2dToTfjs(this.backend, w.weight);
    const y = this.backend.conv2d(x, k, [2, 2], 'same');
    this.backend.dispose(k);
    const s = this.backend.getShape(y);
    const bias = this.backend.reshape(w.bias, [1, 1, 1, s[3]!]);
    const z = this.backend.add(y, bias);
    this.backend.dispose(y);
    this.backend.dispose(bias);
    const r = this.backend.relu(z);
    this.backend.dispose(z);
    this.backend.dispose(x);
    return r;
  }
}