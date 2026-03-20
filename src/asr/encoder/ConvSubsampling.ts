import type { ComputeBackend } from '../compute/Backend';
import type { SubsamplingWeights } from '../model/weights';
import type { TensorHandle } from '../compute/types';
import { linearForward } from './ops';

export class ConvSubsampling {
  constructor(
    private readonly backend: ComputeBackend,
    private readonly w: SubsamplingWeights,
  ) {}

  forward(melFeatures: TensorHandle): TensorHandle {
    const b = this.backend;
    let x = melFeatures;
    const shape = b.getShape(x);
    const B = shape[0];
    const T = shape[1];
    const F = shape[2];
    x = b.reshape(x, [B, T, F, 1]);
    x = b.transpose(x, [0, 3, 1, 2]);

    for (const layer of this.w.convLayers) {
      const tfF = b.transpose(layer.weight, [2, 3, 1, 0]);
      x = b.conv2d(x, tfF, [2, 2], 'same');
      b.dispose(tfF);
      if (layer.bias) {
        const biased = b.add(x, b.reshape(layer.bias, [1, 1, 1, -1]));
        b.dispose(x);
        x = biased;
      }
      x = b.relu(x);
    }

    const convShape = b.getShape(x);
    const tOut = convShape[2];
    const fOut = convShape[3];
    const cOut = convShape[1];
    const flat = b.reshape(b.transpose(x, [0, 2, 1, 3]), [B, tOut, cOut * fOut]);
    b.dispose(x);
    const out = linearForward(b, flat, this.w.out.weight, this.w.out.bias);
    b.dispose(flat);
    return out;
  }
}
