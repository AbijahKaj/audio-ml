import { ComputeScope } from '../compute/Scope.js';
import type { ComputeBackend } from '../compute/Backend.js';
import type { TensorHandle } from '../compute/types.js';
import type { ConvModuleWeights } from '../model/WeightMapper.js';
import { Linear } from './Linear.js';

export class ConvModule {
  private readonly pointwiseIn: Linear;
  private readonly pointwiseOut: Linear;

  constructor(
    private readonly backend: ComputeBackend,
    private readonly weights: ConvModuleWeights,
    private readonly padding: number,
  ) {
    this.pointwiseIn = new Linear(backend, weights.pointwiseIn);
    this.pointwiseOut = new Linear(backend, weights.pointwiseOut);
  }

  forward(input: TensorHandle): TensorHandle {
    if (!this.weights.pointwiseIn.weight || !this.weights.pointwiseOut.weight) {
      return input;
    }

    const scope = new ComputeScope();
    let hidden = this.weights.norm.weight && this.weights.norm.bias
      ? scope.track(this.backend.layerNorm(input, this.weights.norm.weight, this.weights.norm.bias, 1e-5))
      : input;

    hidden = scope.track(this.pointwiseIn.forward(hidden));
    const [value, gate] = this.backend.split(hidden, 2, -1);
    scope.track(value);
    scope.track(gate);
    const gated = scope.track(this.backend.mul(value, this.backend.sigmoid(gate)));

    let convolved = gated;
    if (this.weights.depthwiseWeight) {
      convolved = scope.track(this.backend.depthwiseConv1d(gated, this.weights.depthwiseWeight, 1, this.padding));
    }

    if (
      this.weights.batchNormMean &&
      this.weights.batchNormVariance &&
      this.weights.batchNormScale &&
      this.weights.batchNormOffset
    ) {
      convolved = scope.track(this.backend.batchNorm(
        convolved,
        this.weights.batchNormMean,
        this.weights.batchNormVariance,
        this.weights.batchNormScale,
        this.weights.batchNormOffset,
        1e-5,
      ));
    }

    const activated = scope.track(this.backend.silu(convolved));
    const output = this.pointwiseOut.forward(activated);
    scope.dispose(this.backend);
    return output;
  }
}
