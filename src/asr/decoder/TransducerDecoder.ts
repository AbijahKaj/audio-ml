import type { ComputeBackend } from '../compute/Backend';
import type { TensorHandle } from '../compute/types';
import type { DecoderState, DecodeResult } from './types';
import { PredictionNetwork } from './PredictionNetwork';

export abstract class TransducerDecoder {
  protected readonly blankId: number;
  protected readonly maxSymbolsPerStep: number;

  constructor(
    protected backend: ComputeBackend,
    protected prediction: PredictionNetwork,
    blankId: number,
    maxSymbolsPerStep = 10,
  ) {
    this.blankId = blankId;
    this.maxSymbolsPerStep = maxSymbolsPerStep;
  }

  initialState(): DecoderState {
    const state = this.prediction.initialState();
    return {
      ...state,
      lastToken: this.blankId,
      frameOffset: 0,
    };
  }

  disposeState(state: DecoderState): void {
    this.backend.dispose(state.h);
    this.backend.dispose(state.c);
  }

  abstract decode(encoderOutput: TensorHandle, state?: DecoderState): Promise<DecodeResult>;

  protected async argmax(logits: TensorHandle): Promise<number> {
    const data = await this.backend.getData(logits);
    let bestIndex = 0;
    let bestValue = -Infinity;
    for (let i = 0; i < data.length; i++) {
      if (data[i] > bestValue) {
        bestValue = data[i];
        bestIndex = i;
      }
    }
    return bestIndex;
  }

  protected frameAt(encoderOutput: TensorHandle, frame: number): TensorHandle {
    const shape = this.backend.getShape(encoderOutput);
    return this.backend.slice(encoderOutput, [0, frame, 0], [shape[0], 1, shape[2]]);
  }
}
