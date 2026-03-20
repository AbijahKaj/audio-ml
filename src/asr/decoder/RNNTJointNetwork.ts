import type { TensorHandle } from '../compute/types';
import type { ComputeBackend } from '../compute/Backend';
import type { JointWeights } from '../model/ModelWeights';
import { Linear } from '../encoder/Linear';
import { JointNetwork } from './JointNetwork';

export class RNNTJointNetwork extends JointNetwork {
  private readonly outputProj: Linear;

  constructor(backend: ComputeBackend, weights: JointWeights) {
    super(backend, weights);
    this.outputProj = new Linear(backend, weights.tokenProj);
  }

  forward(encoderFrame: TensorHandle, predictionOut: TensorHandle): { tokenLogits: TensorHandle } {
    const joint = this.computeJoint(encoderFrame, predictionOut);
    const tokenLogits = this.outputProj.forward(joint);
    this.backend.dispose(joint);
    return { tokenLogits };
  }
}
