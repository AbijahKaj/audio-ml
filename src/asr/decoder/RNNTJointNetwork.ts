import type { ComputeBackend } from '../compute/Backend';
import type { JointWeights } from '../model/weights';
import type { TensorHandle } from '../compute/types';
import { linearForward } from '../encoder/ops';

export class RNNTJointNetwork {
  constructor(
    private readonly backend: ComputeBackend,
    private readonly w: JointWeights,
  ) {}

  forward(
    encoderFrame: TensorHandle,
    predictionOut: TensorHandle,
  ): { tokenLogits: TensorHandle } {
    const b = this.backend;
    const fe = linearForward(b, encoderFrame, this.w.enc.weight, this.w.enc.bias);
    const fg = linearForward(b, predictionOut, this.w.pred.weight, this.w.pred.bias);
    const joint = b.relu(b.add(fe, fg));
    b.dispose(fe);
    b.dispose(fg);
    const tokenLogits = linearForward(b, joint, this.w.jointNetLinear.weight, this.w.jointNetLinear.bias);
    b.dispose(joint);
    return { tokenLogits };
  }
}
