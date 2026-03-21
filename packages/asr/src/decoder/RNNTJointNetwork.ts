import type { ComputeBackend } from '../compute/Backend';
import type { TensorHandle } from '../compute/types';
import type { JointNetworkWeights } from '../model/WeightMapper';
import { JointNetwork } from './JointNetwork';
import { Linear } from '../encoder/Linear';

export class RNNTJointNetwork extends JointNetwork {
  private outputProj: Linear;

  constructor(backend: ComputeBackend, weights: JointNetworkWeights) {
    super(backend, weights);
    this.outputProj = new Linear(backend, weights.outputProj);
  }

  forward(
    encoderFrame: TensorHandle,
    predictionOut: TensorHandle
  ): { tokenLogits: TensorHandle } {
    const joint = this.computeJoint(encoderFrame, predictionOut);
    const tokenLogits = this.outputProj.forward(joint);
    this.backend.dispose(joint);
    return { tokenLogits };
  }
}
