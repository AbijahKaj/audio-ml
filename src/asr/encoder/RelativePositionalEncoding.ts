import type { ComputeBackend } from '../compute/Backend';
import type { TensorHandle } from '../compute/types';

export class RelativePositionalEncoding {
  constructor(private readonly backend: ComputeBackend) {}

  forward(query: TensorHandle, timeSteps: number): TensorHandle {
    const shape = this.backend.getShape(query);
    return this.backend.zeros([shape[0], shape[1], shape[2], timeSteps]);
  }
}
