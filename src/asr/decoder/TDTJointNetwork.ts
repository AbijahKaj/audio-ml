import type { TensorHandle } from '../compute/types';
import type { ComputeBackend } from '../compute/Backend';
import type { JointWeights } from '../model/ModelWeights';
import { Linear } from '../encoder/Linear';
import { JointNetwork } from './JointNetwork';

export class TDTJointNetwork extends JointNetwork {
  private readonly tokenProj: Linear;
  private readonly durationProj: Linear;

  constructor(backend: ComputeBackend, weights: JointWeights) {
    super(backend, weights);
    if (!weights.durationProj) {
      throw new Error('TDT joint network requires durationProj weights.');
    }
    this.tokenProj = new Linear(backend, weights.tokenProj);
    this.durationProj = new Linear(backend, weights.durationProj);
  }

  forward(
    encoderFrame: TensorHandle,
    predictionOut: TensorHandle,
  ): { tokenLogits: TensorHandle; durationLogits: TensorHandle } {
    const joint = this.computeJoint(encoderFrame, predictionOut);
    const tokenLogits = this.tokenProj.forward(joint);
    const durationLogits = this.durationProj.forward(joint);
    this.backend.dispose(joint);
    return { tokenLogits, durationLogits };
  }
}
