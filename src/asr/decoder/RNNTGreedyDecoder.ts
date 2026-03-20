import type { ComputeBackend } from '../compute/Backend';
import type { TensorHandle } from '../compute/types';
import { PredictionNetwork } from './PredictionNetwork';
import { RNNTJointNetwork } from './RNNTJointNetwork';
import { TransducerDecoder } from './TransducerDecoder';
import type { DecoderState, DecodeResult } from './types';

export class RNNTGreedyDecoder extends TransducerDecoder {
  constructor(
    backend: ComputeBackend,
    prediction: PredictionNetwork,
    private joint: RNNTJointNetwork,
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

      while (symbolsEmitted < this.maxSymbolsPerStep) {
        const step = this.prediction.step(lastToken, h, c);
        const { tokenLogits } = this.joint.forward(encoderFrame, step.output);
        const token = await this.argmax(tokenLogits);

        this.backend.dispose(tokenLogits);
        this.backend.dispose(step.output);

        if (token === this.blankId) {
          this.backend.dispose(step.h);
          this.backend.dispose(step.c);
          break;
        }

        tokenIds.push(token);
        this.backend.dispose(h);
        this.backend.dispose(c);
        h = step.h;
        c = step.c;
        lastToken = token;
        symbolsEmitted += 1;
      }

      this.backend.dispose(encoderFrame);
      frame += 1;
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
