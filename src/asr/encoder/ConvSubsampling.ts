import type { ComputeBackend } from '../compute/Backend';
import type { SubsamplingWeights } from '../model/weights';
import type { TensorHandle } from '../compute/types';
import { linearForward } from './ops';

/**
 * NeMo: preprocessor [B, F, T] → encoder transposes to [B, T, F] → unsqueeze(1) → [B, 1, T, F].
 * TF NHWC matches that 2D grid: height = time, width = mel.
 */
export class ConvSubsampling {
  constructor(
    private readonly backend: ComputeBackend,
    private readonly w: SubsamplingWeights,
  ) {}

  forward(melFeatures: TensorHandle): TensorHandle {
    const b = this.backend;
    const shape = b.getShape(melFeatures);
    const B = shape[0];
    const T = shape[1];
    const F = shape[2];
    let x = b.reshape(melFeatures, [B, T, F, 1]);

    for (const layer of this.w.layers) {
      if (layer.kind === 'conv2d_s2') {
        const tfF = b.transpose(layer.weight, [2, 3, 1, 0]);
        const y = b.conv2d(x, tfF, [2, 2], 'same');
        b.dispose(tfF);
        b.dispose(x);
        x = y;
      } else if (layer.kind === 'depthwise_s2') {
        const sh = b.getShape(layer.weight);
        const cOut = sh[0]!;
        const k = sh[2]!;
        const flatW = b.reshape(layer.weight, [cOut, k * k]);
        const tfF = b.reshape(flatW, [k, k, cOut, 1]);
        b.dispose(flatW);
        const y = b.depthwiseConv2d(x, tfF, [2, 2], 'same');
        b.dispose(tfF);
        b.dispose(x);
        x = y;
      } else {
        const tfF = b.transpose(layer.weight, [2, 3, 1, 0]);
        const y = b.conv2d(x, tfF, [1, 1], 'same');
        b.dispose(tfF);
        b.dispose(x);
        x = y;
      }

      if (layer.bias) {
        const biased = b.add(x, b.reshape(layer.bias, [1, 1, 1, -1]));
        b.dispose(x);
        x = biased;
      }
      const reluOut = b.relu(x);
      b.dispose(x);
      x = reluOut;
    }

    const convShape = b.getShape(x);
    const H = convShape[1]!;
    const W = convShape[2]!;
    const C = convShape[3]!;
    const xHcw = b.transpose(x, [0, 1, 3, 2]);
    b.dispose(x);
    const flat = b.reshape(xHcw, [B, H, C * W]);
    b.dispose(xHcw);
    const out = linearForward(b, flat, this.w.out.weight, this.w.out.bias);
    b.dispose(flat);
    return out;
  }
}
