import type { ComputeBackend } from '../compute/Backend';
import type { TensorHandle } from '../compute/types';
import type { RNNTJointWeights } from '../model/types';
import { linearForward } from '../encoder/linear';

export class RNNTJointNetwork {
  constructor(
    private readonly backend: ComputeBackend,
    private readonly w: RNNTJointWeights,
  ) {}

  forward(encoderFrame: TensorHandle, predictionHidden: TensorHandle): TensorHandle {
    const enc = linearForward(this.backend, encoderFrame, this.w.encoderProj);
    const pred = linearForward(this.backend, predictionHidden, this.w.predProj);
    const sum = this.backend.add(enc, pred);
    this.backend.dispose(enc);
    this.backend.dispose(pred);
    const act = this.backend.relu(sum);
    this.backend.dispose(sum);
    const logits = linearForward(this.backend, act, this.w.output);
    this.backend.dispose(act);
    return logits;
  }
}