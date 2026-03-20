import type { ComputeBackend } from '../compute/Backend.js';
import type { TensorHandle } from '../compute/types.js';
import type { FastConformerConfig } from '../model/ModelConfig.js';
import { PredictionNetwork } from './PredictionNetwork.js';
import { RNNTJointNetwork } from './RNNTJointNetwork.js';
import { TransducerDecoder } from './TransducerDecoder.js';

export class RNNTGreedyDecoder extends TransducerDecoder {
  constructor(
    backend: ComputeBackend,
    config: FastConformerConfig,
    private readonly predictionNetwork: PredictionNetwork,
    private readonly jointNetwork: RNNTJointNetwork,
  ) {
    super(backend, config);
  }

  async decode(encoderOutput: TensorHandle): Promise<number[]> {
    const [batch, time, features] = this.backend.getShape(encoderOutput);
    const tokens: number[] = [];
    let state = this.predictionNetwork.initialState();
    let lastToken = this.blankId;

    for (let frameIndex = 0; frameIndex < time; frameIndex += 1) {
      const encoderFrame = this.backend.slice(encoderOutput, [0, frameIndex, 0], [batch, 1, features]);

      for (let symbolIndex = 0; symbolIndex < this.maxSymbolsPerStep; symbolIndex += 1) {
        const next = this.predictionNetwork.step(lastToken, state);
        const logits = this.jointNetwork.forward(encoderFrame, next.output);
        const token = await this.argMax(logits.tokenLogits);
        this.backend.dispose(logits.tokenLogits);
        if (next.output !== next.h && next.output !== next.c) {
          this.backend.dispose(next.output);
        }

        if (token === this.blankId) {
          if (next.h !== state.h) {
            this.backend.dispose(next.h);
          }
          if (next.c !== state.c) {
            this.backend.dispose(next.c);
          }
          break;
        }

        if (state.h !== next.h) {
          this.backend.dispose(state.h);
        }
        if (state.c !== next.c) {
          this.backend.dispose(state.c);
        }

        state = { h: next.h, c: next.c };
        lastToken = token;
        tokens.push(token);
      }

      this.backend.dispose(encoderFrame);
    }

    this.backend.dispose(state.h);
    this.backend.dispose(state.c);
    return tokens;
  }
}
