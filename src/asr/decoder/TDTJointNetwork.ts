import { ComputeScope } from '../compute/index.js';
import type { TensorHandle } from '../compute/index.js';
import type { JointNetworkWeights } from '../model/WeightMapper.js';
import type { ComputeBackend } from '../compute/Backend.js';
import { JointNetwork } from './JointNetwork.js';
import { Linear } from '../encoder/Linear.js';

/**
 * TDT (Token-and-Duration Transducer) Joint Network.
 *
 * Outputs:
 *   - token logits     [vocab_size]
 *   - duration logits  [num_durations]  — how many encoder frames to skip
 *
 * Reference: Xu et al. (ICML 2023) — "Efficient Sequence Transduction
 * by Jointly Predicting Tokens and Durations"
 */
export class TDTJointNetwork extends JointNetwork {
  private readonly tokenProj: Linear;
  private readonly durationProj: Linear;

  constructor(backend: ComputeBackend, weights: JointNetworkWeights) {
    super(backend, weights);
    if (!weights.durationProj) {
      throw new Error('TDTJointNetwork requires durationProj weights');
    }
    this.tokenProj = Linear.fromWeights(backend, weights.tokenProj);
    this.durationProj = Linear.fromWeights(backend, weights.durationProj);
  }

  forward(
    encoderFrame: TensorHandle,
    predictionOut: TensorHandle,
  ): { tokenLogits: TensorHandle; durationLogits: TensorHandle } {
    const scope = new ComputeScope();
    const joint = scope.track(this.computeJoint(encoderFrame, predictionOut));
    const tokenLogits = this.tokenProj.forward(joint);
    const durationLogits = this.durationProj.forward(joint);
    scope.dispose(this.backend);
    return { tokenLogits, durationLogits };
  }
}
