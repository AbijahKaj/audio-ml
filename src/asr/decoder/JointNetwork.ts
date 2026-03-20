import { ComputeScope } from '../compute/ComputeScope';
import type { ComputeBackend } from '../compute/Backend';
import type { TensorHandle } from '../compute/types';
import type { JointWeights } from '../model/ModelWeights';
import { Linear } from '../encoder/Linear';

export class JointNetwork {
  protected readonly encoderProj: Linear;
  protected readonly predictionProj: Linear;
  protected readonly hiddenProj?: Linear;

  constructor(
    protected backend: ComputeBackend,
    protected weights: JointWeights,
  ) {
    this.encoderProj = new Linear(backend, weights.encoderProj);
    this.predictionProj = new Linear(backend, weights.predictionProj);
    this.hiddenProj = weights.hiddenProj ? new Linear(backend, weights.hiddenProj) : undefined;
  }

  protected computeJoint(encoderFrame: TensorHandle, predictionOut: TensorHandle): TensorHandle {
    const scope = new ComputeScope();
    const predShape = this.backend.getShape(predictionOut);
    const pred3d = predShape.length === 2
      ? scope.track(this.backend.reshape(predictionOut, [predShape[0], 1, predShape[1]]))
      : predictionOut;

    const enc = scope.track(this.encoderProj.forward(encoderFrame));
    const pred = scope.track(this.predictionProj.forward(pred3d));
    const added = scope.track(this.backend.add(enc, pred));
    const activated = scope.track(this.backend.relu(added));
    const joint = this.hiddenProj ? this.hiddenProj.forward(activated) : activated;

    scope.keep(joint);
    scope.dispose(this.backend);
    return joint;
  }
}
