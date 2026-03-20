import type { ComputeBackend } from '../compute/Backend';
import type { TensorHandle } from '../compute/types';
import type { TDTJointWeights } from '../model/types';
import { linearForward } from '../encoder/linear';

export class TDTJointNetwork {
  constructor(
    private readonly backend: ComputeBackend,
    private readonly w: TDTJointWeights,
  ) {}

  forward(
    encoderFrame: TensorHandle,
    predictionHidden: TensorHandle,
  ): { tokenLogits: TensorHandle; durationLogits: TensorHandle } {
    const enc = linearForward(this.backend, encoderFrame, this.w.encoderProj);
    const pred = linearForward(this.backend, predictionHidden, this.w.predProj);
    const sum = this.backend.add(enc, pred);
    this.backend.dispose(enc);
    this.backend.dispose(pred);
    const act = this.backend.relu(sum);
    this.backend.dispose(sum);
    const tokenLogits = linearForward(this.backend, act, this.w.output);
    const durationLogits = linearForward(this.backend, act, this.w.duration);
    this.backend.dispose(act);
    return { tokenLogits, durationLogits };
  }
}