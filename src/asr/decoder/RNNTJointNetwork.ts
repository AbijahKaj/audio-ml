import type { TensorHandle } from '../compute/types.js';
import type { ComputeBackend } from '../compute/Backend.js';
import type { JointWeights } from '../model/WeightMapper.js';
import { Linear } from '../encoder/Linear.js';
import { JointNetwork } from './JointNetwork.js';

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
