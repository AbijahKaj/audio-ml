import type { ComputeBackend } from '../compute/Backend.js';
import type { TensorHandle } from '../compute/types.js';
import type { FastConformerConfig } from '../model/ModelConfig.js';
import { PredictionNetwork } from './PredictionNetwork.js';
import { TDTJointNetwork } from './TDTJointNetwork.js';
import { TransducerDecoder } from './TransducerDecoder.js';

export class TDTGreedyDecoder extends TransducerDecoder {
  private readonly durations: number[];

  constructor(
    backend: ComputeBackend,
    config: FastConformerConfig,
    private readonly predictionNetwork: PredictionNetwork,
    private readonly jointNetwork: TDTJointNetwork,
  ) {
    super(backend, config);
    this.durations = config.tdtDurations ?? Array.from({ length: config.tdtNumDurations ?? 5 }, (_, index) => index);
  }

  async decode(encoderOutput: TensorHandle): Promise<number[]> {
    const [batch, time, features] = this.backend.getShape(encoderOutput);
    const tokens: number[] = [];
    let state = this.predictionNetwork.initialState();
    let lastToken = this.blankId;
    let frameIndex = 0;

    while (frameIndex < time) {
      const encoderFrame = this.backend.slice(encoderOutput, [0, frameIndex, 0], [batch, 1, features]);
      let advanced = false;

      for (let symbolIndex = 0; symbolIndex < this.maxSymbolsPerStep; symbolIndex += 1) {
        const next = this.predictionNetwork.step(lastToken, state);
        const logits = this.jointNetwork.forward(encoderFrame, next.output);
        const token = await this.argMax(logits.tokenLogits);
        const durationIndex = await this.argMax(logits.durationLogits);
        const duration = this.durations[durationIndex] ?? 1;
        this.backend.dispose(logits.tokenLogits);
        this.backend.dispose(logits.durationLogits);
        if (next.output !== next.h && next.output !== next.c) {
          this.backend.dispose(next.output);
        }

        const advanceBy = Math.max(1, duration);

        if (token === this.blankId) {
          if (next.h !== state.h) {
            this.backend.dispose(next.h);
          }
          if (next.c !== state.c) {
            this.backend.dispose(next.c);
          }
          frameIndex += advanceBy;
          advanced = true;
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

        if (advanceBy > 0) {
          frameIndex += advanceBy;
          advanced = true;
          break;
        }
      }

      if (!advanced) {
        frameIndex += 1;
      }

      this.backend.dispose(encoderFrame);
    }

    this.backend.dispose(state.h);
    this.backend.dispose(state.c);
    return tokens;
  }
}
