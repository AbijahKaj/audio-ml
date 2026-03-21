import type { ComputeBackend } from '../compute/Backend';
import type { TensorHandle } from '../compute/types';
import type { JointNetworkWeights } from '../model/WeightMapper';
import { Linear } from '../encoder/Linear';

/**
 * Base joint network that combines encoder and prediction outputs.
 * Shared infrastructure for both RNNT and TDT variants.
 */
export class JointNetwork {
  protected backend: ComputeBackend;
  protected encoderProj: Linear;
  protected predictionProj: Linear;

  constructor(backend: ComputeBackend, weights: JointNetworkWeights) {
    this.backend = backend;
    this.encoderProj = new Linear(backend, weights.encoderProj);
    this.predictionProj = new Linear(backend, weights.predictionProj);
  }

  protected computeJoint(encoderFrame: TensorHandle, predictionOut: TensorHandle): TensorHandle {
    return this.backend.tidy(() => {
      const enc = this.encoderProj.forward(encoderFrame);
      const pred = this.predictionProj.forward(predictionOut);
      return this.backend.relu(this.backend.add(enc, pred));
    });
  }
}
