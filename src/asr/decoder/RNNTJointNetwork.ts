import { ComputeScope } from '../compute/index.js';
import type { TensorHandle } from '../compute/index.js';
import type { JointNetworkWeights } from '../model/WeightMapper.js';
import type { ComputeBackend } from '../compute/Backend.js';
import { JointNetwork } from './JointNetwork.js';
import { Linear } from '../encoder/Linear.js';

/**
 * RNNT Joint Network.
 *
 * Output: token logits [vocab_size]
 *
 * joint_input → Linear → [vocab_size]
 */
export class RNNTJointNetwork extends JointNetwork {
  private readonly tokenProj: Linear;

  constructor(backend: ComputeBackend, weights: JointNetworkWeights) {
    super(backend, weights);
    this.tokenProj = Linear.fromWeights(backend, weights.tokenProj);
  }

  forward(
    encoderFrame: TensorHandle,
    predictionOut: TensorHandle,
  ): { tokenLogits: TensorHandle } {
    const scope = new ComputeScope();
    const joint = scope.track(this.computeJoint(encoderFrame, predictionOut));
    const tokenLogits = this.tokenProj.forward(joint);
    scope.dispose(this.backend);
    return { tokenLogits };
  }
}
