import type { ComputeBackend } from '../compute/Backend';
import type { FastConformerConfig } from '../model/ModelConfig';
import type { TensorHandle } from '../compute/types';
import { PredictionNetwork } from './PredictionNetwork';
import { TDTJointNetwork } from './TDTJointNetwork';

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

export class TDTGreedyDecoder {
  private readonly maxSymbolsPerStep = 10;
  private readonly durations: number[];

  constructor(
    private readonly backend: ComputeBackend,
    private readonly predNet: PredictionNetwork,
    private readonly jointNet: TDTJointNetwork,
    private readonly blankId: number,
    config: FastConformerConfig,
  ) {
    this.durations = config.tdtDurations ?? [0, 1, 2, 3, 4];
  }

  async decode(encoderOutput: TensorHandle): Promise<number[]> {
    const b = this.backend;
    const T = b.getShape(encoderOutput)[1];
    const tokens: number[] = [];
    let { h, c } = this.predNet.initialState();
    let lastToken = this.blankId;
    let t = 0;

    while (t < T) {
      const encFrame = b.slice(encoderOutput, [0, t, 0], [1, 1, -1]);
      let symbolsEmitted = 0;

      while (symbolsEmitted < this.maxSymbolsPerStep) {
        const { output: predOut, h: hNew, c: cNew } = this.predNet.step(lastToken, h, c);
        const { tokenLogits, durationLogits } = this.jointNet.forward(encFrame, predOut);

        const tokenData = (await b.getData(tokenLogits)) as Float32Array;
        const durData = (await b.getData(durationLogits)) as Float32Array;
        const token = argmax(tokenData);
        const durIdx = argmax(durData);
        const duration = this.durations[durIdx] ?? 0;

        b.dispose(tokenLogits);
        b.dispose(durationLogits);
        b.dispose(predOut);

        if (token === this.blankId) {
          b.dispose(hNew);
          b.dispose(cNew);
          t += Math.max(1, duration);
          break;
        }

        tokens.push(token);
        b.dispose(h);
        b.dispose(c);
        h = hNew;
        c = cNew;
        lastToken = token;
        symbolsEmitted++;
        t += duration;
        if (duration > 0) {
          break;
        }
      }

      b.dispose(encFrame);
    }

    b.dispose(h);
    b.dispose(c);
    return tokens;
  }
}
