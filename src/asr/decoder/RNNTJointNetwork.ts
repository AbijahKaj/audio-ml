import type { TensorHandle } from '../compute/types';
import type { ComputeBackend } from '../compute/Backend';
import type { JointWeights } from '../model/WeightMapper';
import { Linear } from '../encoder/Linear';
import { JointNetwork } from './JointNetwork';

export class RNNTJointNetwork extends JointNetwork {
  private readonly outputProjection: Linear;

  constructor(backend: ComputeBackend, weights: JointWeights) {
    super(backend, weights);
    this.outputProjection = new Linear(backend, weights.outputProj);
  }

  forward(encoderFrame: TensorHandle, predictionOut: TensorHandle): { tokenLogits: TensorHandle } {
    const joint = this.computeJoint(encoderFrame, predictionOut);
    const tokenLogits = this.outputProjection.forward(joint);
    if (tokenLogits !== joint) {
      this.backend.dispose(joint);
    }
    return { tokenLogits };
  }
}
