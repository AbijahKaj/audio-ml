import type { ComputeBackend, TensorHandle } from '../compute/index.js';
import { ComputeScope } from '../compute/index.js';
import type { JointNetworkWeights } from '../model/WeightMapper.js';
import { Linear } from '../encoder/Linear.js';

/**
 * Base joint network — projects encoder and prediction outputs into a shared space.
 *
 * joint_input = relu(encoder_proj(enc_frame) + prediction_proj(pred_out))
 *
 * Both RNNT and TDT extend this with different output heads.
 */
export abstract class JointNetwork {
  protected readonly encoderProj: Linear;
  protected readonly predictionProj: Linear;

  constructor(
    protected readonly backend: ComputeBackend,
    weights: JointNetworkWeights,
  ) {
    this.encoderProj = Linear.fromWeights(backend, weights.encoderProj);
    this.predictionProj = Linear.fromWeights(backend, weights.predictionProj);
  }

  protected computeJoint(
    encoderFrame: TensorHandle,
    predictionOut: TensorHandle,
  ): TensorHandle {
    const scope = new ComputeScope();
    const enc = scope.track(this.encoderProj.forward(encoderFrame));
    const pred = scope.track(this.predictionProj.forward(predictionOut));
    const sum = scope.track(this.backend.add(enc, pred));
    const out = this.backend.relu(sum);
    scope.dispose(this.backend);
    return out;
  }
}
