import type { ComputeBackend } from '../compute/Backend';
import type { TensorHandle } from '../compute/types';
import type { JointWeights } from '../model/WeightMapper';
import { Linear } from '../encoder/Linear';

export class JointNetwork {
  protected readonly encoderProjection: Linear;
  protected readonly predictionProjection: Linear;

  constructor(
    protected readonly backend: ComputeBackend,
    protected readonly weights: JointWeights,
  ) {
    this.encoderProjection = new Linear(backend, weights.encoderProj);
    this.predictionProjection = new Linear(backend, weights.predictionProj);
  }

  protected computeJoint(encoderFrame: TensorHandle, predictionOut: TensorHandle): TensorHandle {
    const encoderPrepared = this.ensure3d(encoderFrame);
    const predictionPrepared = this.ensure3d(predictionOut);
    const projectedEncoder = this.encoderProjection.forward(encoderPrepared);
    const projectedPrediction = this.predictionProjection.forward(predictionPrepared);
    const summed = this.backend.add(projectedEncoder, projectedPrediction);
    if (projectedEncoder !== encoderPrepared) {
      this.backend.dispose(projectedEncoder);
    }
    if (projectedPrediction !== predictionPrepared) {
      this.backend.dispose(projectedPrediction);
    }
    if (encoderPrepared !== encoderFrame) {
      this.backend.dispose(encoderPrepared);
    }
    if (predictionPrepared !== predictionOut) {
      this.backend.dispose(predictionPrepared);
    }
    return this.backend.relu(summed);
  }

  private ensure3d(tensor: TensorHandle): TensorHandle {
    const shape = this.backend.getShape(tensor);
    if (shape.length === 3) {
      return tensor;
    }
    if (shape.length === 2) {
      return this.backend.reshape(tensor, [shape[0], 1, shape[1]]);
    }
    if (shape.length === 1) {
      return this.backend.reshape(tensor, [1, 1, shape[0]]);
    }
    return tensor;
  }
}
