import type { ComputeBackend } from '../compute/Backend';
import type { TensorHandle } from '../compute/types';
import { PredictionNetwork, type PredictionState } from './PredictionNetwork';
import { TDTJointNetwork } from './TDTJointNetwork';

function argmax(data: Float32Array): number {
  let maxIdx = 0;
  let maxVal = data[0];
  for (let i = 1; i < data.length; i++) {
    if (data[i] > maxVal) {
      maxVal = data[i];
      maxIdx = i;
    }
  }
  return maxIdx;
}

/**
 * TDT (Token-and-Duration Transducer) greedy decoder.
 * Unlike RNNT, TDT predicts both a token and a duration (frame skip count),
 * enabling 2-5x faster inference by skipping frames.
 */
export class TDTGreedyDecoder {
  private backend: ComputeBackend;
  private predNet: PredictionNetwork;
  private jointNet: TDTJointNetwork;
  private blankId: number;
  private durations: number[];
  private maxSymbolsPerStep: number;

  constructor(
    backend: ComputeBackend,
    predNet: PredictionNetwork,
    jointNet: TDTJointNetwork,
    durations: number[] = [0, 1, 2, 3, 4],
    blankId: number = 0,
    maxSymbolsPerStep: number = 10,
  ) {
    this.backend = backend;
    this.predNet = predNet;
    this.jointNet = jointNet;
    this.durations = durations;
    this.blankId = blankId;
    this.maxSymbolsPerStep = maxSymbolsPerStep;
  }

  async decode(encoderOutput: TensorHandle): Promise<number[]> {
    const T = this.backend.getShape(encoderOutput)[1] as number;
    const D = this.backend.getShape(encoderOutput)[2] as number;
    const tokens: number[] = [];
    let state = this.predNet.initialState();
    let lastToken = this.blankId;
    let t = 0;

    while (t < T) {
      const encFrame = this.backend.slice(encoderOutput, [0, t, 0], [1, 1, D]);
      let symbolsEmitted = 0;

      while (symbolsEmitted < this.maxSymbolsPerStep) {
        const { output: predOut, newState } = this.predNet.step(lastToken, state);
        const { tokenLogits, durationLogits } = this.jointNet.forward(encFrame, predOut);

        const tokenData = await this.backend.getData(tokenLogits);
        const durData = await this.backend.getData(durationLogits);
        const token = argmax(tokenData);
        const durIdx = argmax(durData);
        const duration = this.durations[durIdx] ?? 1;

        this.backend.dispose(tokenLogits);
        this.backend.dispose(durationLogits);
        this.backend.dispose(predOut);

        if (token === this.blankId) {
          this.predNet.disposeState(newState);
          t += Math.max(1, duration);
          break;
        }

        tokens.push(token);
        this.predNet.disposeState(state);
        state = newState;
        lastToken = token;
        symbolsEmitted++;

        t += duration;
        if (duration > 0) break;
      }

      this.backend.dispose(encFrame);
    }

    this.predNet.disposeState(state);
    return tokens;
  }

  async decodeStreaming(
    encoderOutput: TensorHandle,
    state: PredictionState | null,
    lastToken: number,
    frameOffset: number = 0,
  ): Promise<{
    tokens: number[];
    newState: PredictionState;
    newLastToken: number;
    newFrameOffset: number;
  }> {
    const T = this.backend.getShape(encoderOutput)[1] as number;
    const D = this.backend.getShape(encoderOutput)[2] as number;
    const tokens: number[] = [];
    let currentState = state ?? this.predNet.initialState();
    let currentLastToken = lastToken;
    let t = frameOffset;

    while (t < T) {
      const encFrame = this.backend.slice(encoderOutput, [0, t, 0], [1, 1, D]);
      let symbolsEmitted = 0;

      while (symbolsEmitted < this.maxSymbolsPerStep) {
        const { output: predOut, newState } = this.predNet.step(currentLastToken, currentState);
        const { tokenLogits, durationLogits } = this.jointNet.forward(encFrame, predOut);

        const tokenData = await this.backend.getData(tokenLogits);
        const durData = await this.backend.getData(durationLogits);
        const token = argmax(tokenData);
        const durIdx = argmax(durData);
        const duration = this.durations[durIdx] ?? 1;

        this.backend.dispose(tokenLogits);
        this.backend.dispose(durationLogits);
        this.backend.dispose(predOut);

        if (token === this.blankId) {
          this.predNet.disposeState(newState);
          t += Math.max(1, duration);
          break;
        }

        tokens.push(token);
        this.predNet.disposeState(currentState);
        currentState = newState;
        currentLastToken = token;
        symbolsEmitted++;

        t += duration;
        if (duration > 0) break;
      }

      this.backend.dispose(encFrame);
    }

    return {
      tokens,
      newState: currentState,
      newLastToken: currentLastToken,
      newFrameOffset: Math.max(0, t - T),
    };
  }

  createInitialState(): PredictionState {
    return this.predNet.initialState();
  }

  disposeState(state: PredictionState): void {
    this.predNet.disposeState(state);
  }
}
