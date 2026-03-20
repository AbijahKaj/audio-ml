import { ComputeScope } from '../compute/ComputeScope';
import type { ComputeBackend } from '../compute/Backend';
import type { TensorHandle } from '../compute/types';
import type { PredictionWeights } from '../model/ModelWeights';
import { Linear } from '../encoder/Linear';

export interface PredictionStepOutput {
  output: TensorHandle;
  h: TensorHandle;
  c: TensorHandle;
}

export class PredictionNetwork {
  private readonly outputProj: Linear;
  private readonly hiddenSize: number;

  constructor(
    private backend: ComputeBackend,
    private weights: PredictionWeights,
  ) {
    this.outputProj = new Linear(backend, weights.outputProj);
    this.hiddenSize = this.backend.getShape(weights.lstmWeightHH)[1];
  }

  initialState(): { h: TensorHandle; c: TensorHandle } {
    return {
      h: this.backend.zeros([1, this.hiddenSize]),
      c: this.backend.zeros([1, this.hiddenSize]),
    };
  }

  step(tokenId: number, h: TensorHandle, c: TensorHandle): PredictionStepOutput {
    const scope = new ComputeScope();
    const tokenTensor = scope.track(this.backend.tensor(new Int32Array([tokenId]), [1], 'int32'));
    const embedding = scope.track(this.backend.gather(this.weights.embedding, tokenTensor, 0));

    const wIhT = scope.track(this.backend.transpose(this.weights.lstmWeightIH, [1, 0]));
    const wHhT = scope.track(this.backend.transpose(this.weights.lstmWeightHH, [1, 0]));
    const inputTerm = scope.track(this.backend.matmul(embedding, wIhT));
    const hiddenTerm = scope.track(this.backend.matmul(h, wHhT));
    const bias = scope.track(this.backend.add(this.weights.lstmBiasIH, this.weights.lstmBiasHH));
    const gates = scope.track(this.backend.add(this.backend.add(inputTerm, hiddenTerm), bias));
    const [i, f, g, o] = this.backend.split(gates, 4, -1);
    scope.track(i);
    scope.track(f);
    scope.track(g);
    scope.track(o);

    const iGate = scope.track(this.backend.sigmoid(i));
    const fGate = scope.track(this.backend.sigmoid(f));
    const gGate = scope.track(this.backend.tanh(g));
    const oGate = scope.track(this.backend.sigmoid(o));

    const cCarry = scope.track(this.backend.mul(fGate, c));
    const cInput = scope.track(this.backend.mul(iGate, gGate));
    const cNew = this.backend.add(cCarry, cInput);
    const tanhC = scope.track(this.backend.tanh(cNew));
    const hNew = this.backend.mul(oGate, tanhC);
    const output = this.outputProj.forward(hNew);

    scope.keep(cNew);
    scope.keep(hNew);
    scope.keep(output);
    scope.dispose(this.backend);

    return {
      output,
      h: hNew,
      c: cNew,
    };
  }
}
