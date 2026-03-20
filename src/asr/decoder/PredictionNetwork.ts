import { ComputeScope } from '../compute/Scope.js';
import type { ComputeBackend } from '../compute/Backend.js';
import type { TensorHandle } from '../compute/types.js';
import type { PredictionWeights } from '../model/WeightMapper.js';
import { Linear } from '../encoder/Linear.js';

export interface PredictionState {
  h: TensorHandle;
  c: TensorHandle;
}

export class PredictionNetwork {
  private readonly outputProjection: Linear;

  constructor(
    private readonly backend: ComputeBackend,
    private readonly weights: PredictionWeights,
    private readonly hiddenSize: number,
  ) {
    this.outputProjection = new Linear(backend, weights.outputProj);
  }

  initialState(): PredictionState {
    return {
      h: this.backend.zeros([1, this.hiddenSize]),
      c: this.backend.zeros([1, this.hiddenSize]),
    };
  }

  step(tokenId: number, state: PredictionState): PredictionState & { output: TensorHandle } {
    if (
      !this.weights.embedding ||
      !this.weights.lstmWeightIH ||
      !this.weights.lstmWeightHH ||
      !this.weights.lstmBiasIH ||
      !this.weights.lstmBiasHH
    ) {
      return {
        output: this.outputProjection.forward(state.h),
        h: state.h,
        c: state.c,
      };
    }

    const scope = new ComputeScope();
    const indexTensor = scope.track(this.backend.tensor([tokenId], [1], 'int32'));
    const embedding = scope.track(this.backend.gather(this.weights.embedding, indexTensor, 0));
    const wih = scope.track(this.backend.transpose(this.weights.lstmWeightIH, [1, 0]));
    const whh = scope.track(this.backend.transpose(this.weights.lstmWeightHH, [1, 0]));
    const inputGates = scope.track(this.backend.matmul(embedding, wih));
    const hiddenGates = scope.track(this.backend.matmul(state.h, whh));
    const biasedInput = scope.track(this.backend.add(inputGates, this.weights.lstmBiasIH));
    const biasedHidden = scope.track(this.backend.add(hiddenGates, this.weights.lstmBiasHH));
    const gates = scope.track(this.backend.add(biasedInput, biasedHidden));
    const [inputGate, forgetGate, candidateGate, outputGate] = this.backend.split(gates, 4, -1);
    scope.track(inputGate);
    scope.track(forgetGate);
    scope.track(candidateGate);
    scope.track(outputGate);

    const inputActivated = scope.track(this.backend.sigmoid(inputGate));
    const forgetActivated = scope.track(this.backend.sigmoid(forgetGate));
    const candidateActivated = scope.track(this.backend.tanh(candidateGate));
    const outputActivated = scope.track(this.backend.sigmoid(outputGate));
    const retainedCell = scope.track(this.backend.mul(forgetActivated, state.c));
    const writtenCell = scope.track(this.backend.mul(inputActivated, candidateActivated));
    const cNew = this.backend.add(retainedCell, writtenCell);
    const hNew = this.backend.mul(outputActivated, this.backend.tanh(cNew));
    const output = this.outputProjection.forward(hNew);
    scope.dispose(this.backend);

    return { output, h: hNew, c: cNew };
  }
}
