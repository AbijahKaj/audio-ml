import type { ComputeBackend } from '../compute/Backend';
import type { TensorHandle } from '../compute/types';
import type { FastConformerConfig } from '../model/ModelConfig';
import { PredictionNetwork } from './PredictionNetwork';
import { RNNTJointNetwork } from './RNNTJointNetwork';

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

export class RNNTGreedyDecoder {
  private readonly blankId: number;
  private readonly maxSymbolsPerStep: number;

  constructor(
    private readonly backend: ComputeBackend,
    private readonly predNet: PredictionNetwork,
    private readonly jointNet: RNNTJointNetwork,
    config: FastConformerConfig,
    opts?: { maxSymbolsPerStep?: number },
  ) {
    this.blankId = config.blankTokenId ?? 0;
    this.maxSymbolsPerStep = opts?.maxSymbolsPerStep ?? 10;
  }

  async decode(encoderOutput: TensorHandle): Promise<number[]> {
    const T = this.backend.getShape(encoderOutput)[1]!;
    const tokens: number[] = [];
    let { h, c } = this.predNet.initialState();
    let lastToken = this.blankId;

    for (let t = 0; t < T; t++) {
      const encFrame = this.backend.slice(encoderOutput, [0, t, 0], [1, 1, -1]);
      let symbolsEmitted = 0;

      while (symbolsEmitted < this.maxSymbolsPerStep) {
        const { h: hNew, c: cNew } = this.predNet.step(lastToken, h, c);
        this.backend.dispose(h);
        this.backend.dispose(c);
        h = hNew;
        c = cNew;

        const pred2 = this.backend.reshape(hNew, [1, 1, this.backend.getShape(hNew)[1]!]);
        const logits = this.jointNet.forward(encFrame, pred2);
        this.backend.dispose(pred2);

        const logitsData = await this.backend.getData(logits);
        this.backend.dispose(logits);
        const token = argmax(logitsData);

        if (token === this.blankId) {
          break;
        }

        tokens.push(token);
        lastToken = token;
        symbolsEmitted++;
      }
      this.backend.dispose(encFrame);
    }

    this.backend.dispose(h);
    this.backend.dispose(c);
    return tokens;
  }
}