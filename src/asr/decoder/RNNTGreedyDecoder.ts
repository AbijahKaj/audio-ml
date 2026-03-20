import type { ComputeBackend } from '../compute/Backend';
import type { TensorHandle } from '../compute/types';
import { PredictionNetwork } from './PredictionNetwork';
import { RNNTJointNetwork } from './RNNTJointNetwork';

function argmax(data: Float32Array): number {
  let m = -Infinity;
  let idx = 0;
  for (let i = 0; i < data.length; i++) {
    if (data[i] > m) {
      m = data[i];
      idx = i;
    }
  }
  return idx;
}

export class RNNTGreedyDecoder {
  private readonly maxSymbolsPerStep = 10;

  constructor(
    private readonly backend: ComputeBackend,
    private readonly predNet: PredictionNetwork,
    private readonly jointNet: RNNTJointNetwork,
    private readonly blankId: number,
  ) {}

  async decode(encoderOutput: TensorHandle): Promise<number[]> {
    const b = this.backend;
    const T = b.getShape(encoderOutput)[1];
    const tokens: number[] = [];
    let { h, c } = this.predNet.initialState();
    let lastToken = this.blankId;

    for (let t = 0; t < T; t++) {
      const encFrame = b.slice(encoderOutput, [0, t, 0], [1, 1, -1]);
      let symbolsEmitted = 0;

      while (symbolsEmitted < this.maxSymbolsPerStep) {
        const { output: predOut, h: hNew, c: cNew } = this.predNet.step(lastToken, h, c);
        const { tokenLogits } = this.jointNet.forward(encFrame, predOut);
        const logitsData = (await b.getData(tokenLogits)) as Float32Array;
        const token = argmax(logitsData);

        b.dispose(tokenLogits);
        b.dispose(predOut);

        if (token === this.blankId) {
          b.dispose(hNew);
          b.dispose(cNew);
          break;
        }

        tokens.push(token);
        b.dispose(h);
        b.dispose(c);
        h = hNew;
        c = cNew;
        lastToken = token;
        symbolsEmitted++;
      }

      b.dispose(encFrame);
    }

    b.dispose(h);
    b.dispose(c);
    return tokens;
  }
}
