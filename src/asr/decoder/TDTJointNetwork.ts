import type { ComputeBackend } from '../compute/Backend';
import type { TensorHandle } from '../compute/types';
import type { JointNetworkWeights } from '../model/WeightMapper';
import { JointNetwork } from './JointNetwork';
import { Linear } from '../encoder/Linear';

export class TDTJointNetwork extends JointNetwork {
  private tokenProj: Linear;
  private durationProj: Linear;

  constructor(backend: ComputeBackend, weights: JointNetworkWeights) {
    super(backend, weights);
    this.tokenProj = new Linear(backend, weights.outputProj);

    if (!weights.durationProj) {
      throw new Error('TDT joint network requires duration projection weights');
    }
    this.durationProj = new Linear(backend, weights.durationProj);
  }

  forward(
    encoderFrame: TensorHandle,
    predictionOut: TensorHandle
  ): { tokenLogits: TensorHandle; durationLogits: TensorHandle } {
    const joint = this.computeJoint(encoderFrame, predictionOut);
    const tokenLogits = this.tokenProj.forward(joint);
    const durationLogits = this.durationProj.forward(joint);
    this.backend.dispose(joint);
    return { tokenLogits, durationLogits };
  }
}
