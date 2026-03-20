import type { TensorHandle } from '../compute/types';
import type { ComputeBackend } from '../compute/Backend';
import type { JointWeights } from '../model/WeightMapper';
import { Linear } from '../encoder/Linear';
import { JointNetwork } from './JointNetwork';

export class TDTJointNetwork extends JointNetwork {
  private readonly tokenProjection: Linear;
  private readonly durationProjection: Linear;

  constructor(backend: ComputeBackend, weights: JointWeights) {
    super(backend, weights);
    this.tokenProjection = new Linear(backend, weights.outputProj);
    this.durationProjection = new Linear(backend, weights.durationProj);
  }

  forward(encoderFrame: TensorHandle, predictionOut: TensorHandle): {
    tokenLogits: TensorHandle;
    durationLogits: TensorHandle;
  } {
    const joint = this.computeJoint(encoderFrame, predictionOut);
    const tokenLogits = this.tokenProjection.forward(joint);
    const durationLogits = this.durationProjection.forward(joint);
    if (joint !== tokenLogits && joint !== durationLogits) {
      this.backend.dispose(joint);
    }
    return { tokenLogits, durationLogits };
  }
}
