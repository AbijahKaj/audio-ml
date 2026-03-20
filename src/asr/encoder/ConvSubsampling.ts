import type { ComputeBackend } from '../compute/Backend';
import type { TensorHandle } from '../compute/types';
import type { FastConformerConfig } from '../model/ModelConfig';
import type { EncoderWeights } from '../model/WeightMapper';
import { Linear } from './Linear';

export class ConvSubsampling {
  private readonly inputProjection: Linear;
  private readonly outputProjection: Linear;

  constructor(
    private readonly backend: ComputeBackend,
    weights: EncoderWeights,
    private readonly config: FastConformerConfig,
  ) {
    this.inputProjection = new Linear(backend, weights.subsamplingIn);
    this.outputProjection = new Linear(backend, weights.subsamplingOut);
  }

  forward(melFeatures: TensorHandle): TensorHandle {
    const shape = this.backend.getShape(melFeatures);
    const time = shape[1];
    const stride = Math.max(1, this.config.subsamplingFactor);
    const frames: TensorHandle[] = [];

    for (let index = 0; index < time; index += stride) {
      frames.push(this.backend.slice(melFeatures, [0, index, 0], [shape[0], 1, shape[2]]));
    }

    const downsampled = frames.length > 0 ? this.backend.concat(frames, 1) : melFeatures;
    this.backend.disposeMany(frames);
    const projected = this.inputProjection.forward(downsampled);

    if (projected !== downsampled) {
      this.backend.dispose(downsampled);
    }

    const output = this.outputProjection.forward(projected);
    if (output !== projected) {
      this.backend.dispose(projected);
    }
    return output;
  }
}
