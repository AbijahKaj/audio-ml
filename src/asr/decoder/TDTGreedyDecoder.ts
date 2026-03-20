import type { ComputeBackend } from '../compute/Backend';
import type { TensorHandle } from '../compute/types';
import type { FastConformerConfig } from '../model/ModelConfig';
import { PredictionNetwork } from './PredictionNetwork';
import { TDTJointNetwork } from './TDTJointNetwork';

function argmax(data: Float32Array): number {
  let m = -Infinity;
  let ix = 0;
  for (let i = 0; i < data.length; i++) {
    if (data[i]! > m) {
      m = data[i]!;
      ix = i;
    }
  }
  return ix;
}

export class TDTGreedyDecoder {
  private readonly blankId: number;
  private readonly maxSymbolsPerStep: number;
  private readonly durations: number[];

  constructor(
    private readonly backend: ComputeBackend,
    private readonly predNet: PredictionNetwork,
    private readonly jointNet: TDTJointNetwork,
    config: FastConformerConfig,
    opts?: { maxSymbolsPerStep?: number },
  ) {
    this.blankId = config.blankTokenId ?? 0;
    this.maxSymbolsPerStep = opts?.maxSymbolsPerStep ?? 10;
    this.durations = config.tdtDurations ?? [0, 1, 2, 3, 4];
  }

  async decode(encoderOutput: TensorHandle): Promise<number[]> {
    const T = this.backend.getShape(encoderOutput)[1]!;
    const tokens: number[] = [];
    let { h, c } = this.predNet.initialState();
    let lastToken = this.blankId;
    let t = 0;

    while (t < T) {
      const encFrame = this.backend.slice(encoderOutput, [0, t, 0], [1, 1, -1]);
      let symbolsEmitted = 0;

      while (symbolsEmitted < this.maxSymbolsPerStep) {
        const { h: hNew, c: cNew } = this.predNet.step(lastToken, h, c);
        this.backend.dispose(h);
        this.backend.dispose(c);
        h = hNew;
        c = cNew;

        const pred2 = this.backend.reshape(hNew, [1, 1, this.backend.getShape(hNew)[1]!]);
        const { tokenLogits, durationLogits } = this.jointNet.forward(encFrame, pred2);
        this.backend.dispose(pred2);

        const tokData = await this.backend.getData(tokenLogits);
        const durData = await this.backend.getData(durationLogits);
        this.backend.dispose(tokenLogits);
        this.backend.dispose(durationLogits);

        const token = argmax(tokData);
        const durIdx = argmax(durData);
        const duration = this.durations[durIdx] ?? 0;

        if (token === this.blankId) {
          t += Math.max(1, duration);
          break;
        }

        tokens.push(token);
        lastToken = token;
        symbolsEmitted++;
        t += duration;
        if (duration > 0) {
          break;
        }
      }
      this.backend.dispose(encFrame);
    }

    this.backend.dispose(h);
    this.backend.dispose(c);
    return tokens;
  }
}