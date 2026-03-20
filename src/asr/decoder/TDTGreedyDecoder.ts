import type { ComputeBackend } from '../compute/Backend';
import type { TensorHandle } from '../compute/types';
import { PredictionNetwork } from './PredictionNetwork';
import { TDTJointNetwork } from './TDTJointNetwork';
import { TransducerDecoder } from './TransducerDecoder';
import type { DecoderState, DecodeResult } from './types';

export class TDTGreedyDecoder extends TransducerDecoder {
  constructor(
    backend: ComputeBackend,
    prediction: PredictionNetwork,
    private joint: TDTJointNetwork,
    private durations: number[],
    blankId = 0,
    maxSymbolsPerStep = 10,
  ) {
    super(backend, prediction, blankId, maxSymbolsPerStep);
  }

  async decode(encoderOutput: TensorHandle, state?: DecoderState): Promise<DecodeResult> {
    const initial = state ?? this.initialState();
    let h = initial.h;
    let c = initial.c;
    let lastToken = initial.lastToken;
    let frame = initial.frameOffset;
    const shape = this.backend.getShape(encoderOutput);
    const tokenIds: number[] = [];

    while (frame < shape[1]) {
      const encoderFrame = this.frameAt(encoderOutput, frame);
      let symbolsEmitted = 0;
      let advancedFrame = false;

      while (symbolsEmitted < this.maxSymbolsPerStep) {
        const step = this.prediction.step(lastToken, h, c);
        const { tokenLogits, durationLogits } = this.joint.forward(encoderFrame, step.output);
        const token = await this.argmax(tokenLogits);
        const durationIndex = await this.argmax(durationLogits);
        const duration = this.durations[Math.min(durationIndex, this.durations.length - 1)] ?? 1;

        this.backend.dispose(tokenLogits);
        this.backend.dispose(durationLogits);
        this.backend.dispose(step.output);

        if (token === this.blankId) {
          this.backend.dispose(step.h);
          this.backend.dispose(step.c);
          frame += Math.max(1, duration);
          advancedFrame = true;
          break;
        }

        tokenIds.push(token);
        this.backend.dispose(h);
        this.backend.dispose(c);
        h = step.h;
        c = step.c;
        lastToken = token;
        symbolsEmitted += 1;

        if (duration > 0) {
          frame += duration;
          advancedFrame = true;
          break;
        }
      }

      this.backend.dispose(encoderFrame);
      if (!advancedFrame) {
        frame += 1;
      }
    }

    return {
      tokenIds,
      framesConsumed: frame - initial.frameOffset,
      state: {
        h,
        c,
        lastToken,
        frameOffset: frame,
      },
    };
  }
}
