import { ComputeScope } from '../compute/ComputeScope';
import type { ComputeBackend } from '../compute/Backend';
import type { TensorHandle } from '../compute/types';
import type { LinearWeights } from '../model/ModelWeights';

export class Linear {
  constructor(
    private backend: ComputeBackend,
    private weights: LinearWeights,
  ) {}

  forward(x: TensorHandle): TensorHandle {
    const scope = new ComputeScope();
    const shape = this.backend.getShape(x);
    const weightT = scope.track(this.backend.transpose(this.weights.weight, [1, 0]));
    let output: TensorHandle;

    if (shape.length === 2) {
      output = scope.track(this.backend.matmul(x, weightT));
      if (this.weights.bias) {
        output = scope.track(this.backend.add(output, this.weights.bias));
      }
      scope.keep(output);
      scope.dispose(this.backend);
      return output;
    }

    if (shape.length === 3) {
      const [batch, time, inDim] = shape;
      const outDim = this.backend.getShape(this.weights.weight)[0];
      const flat = scope.track(this.backend.reshape(x, [batch * time, inDim]));
      output = scope.track(this.backend.matmul(flat, weightT));
      if (this.weights.bias) {
        output = scope.track(this.backend.add(output, this.weights.bias));
      }
      const restored = this.backend.reshape(output, [batch, time, outDim]);
      scope.keep(restored);
      scope.dispose(this.backend);
      return restored;
    }

    scope.dispose(this.backend);
    throw new Error(`Linear.forward supports rank-2 or rank-3 tensors. Got rank ${shape.length}.`);
  }
}
