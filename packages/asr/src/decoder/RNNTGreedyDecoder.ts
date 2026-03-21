import type { ComputeBackend } from '../compute/Backend';
import type { TensorHandle } from '../compute/types';
import { PredictionNetwork, type PredictionState } from './PredictionNetwork';
import { RNNTJointNetwork } from './RNNTJointNetwork';

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

export class RNNTGreedyDecoder {
  private backend: ComputeBackend;
  private predNet: PredictionNetwork;
  private jointNet: RNNTJointNetwork;
  private blankId: number;
  private maxSymbolsPerStep: number;

  constructor(
    backend: ComputeBackend,
    predNet: PredictionNetwork,
    jointNet: RNNTJointNetwork,
    blankId: number = 0,
    maxSymbolsPerStep: number = 10,
  ) {
    this.backend = backend;
    this.predNet = predNet;
    this.jointNet = jointNet;
    this.blankId = blankId;
    this.maxSymbolsPerStep = maxSymbolsPerStep;
  }

  async decode(encoderOutput: TensorHandle): Promise<number[]> {
    const T = this.backend.getShape(encoderOutput)[1] as number;
    const D = this.backend.getShape(encoderOutput)[2] as number;
    const tokens: number[] = [];
    let state = this.predNet.initialState();
    let lastToken = this.blankId;

    for (let t = 0; t < T; t++) {
      const encFrame = this.backend.slice(encoderOutput, [0, t, 0], [1, 1, D]);
      let symbolsEmitted = 0;

      while (symbolsEmitted < this.maxSymbolsPerStep) {
        const { output: predOut, newState } = this.predNet.step(lastToken, state);
        const { tokenLogits } = this.jointNet.forward(encFrame, predOut);
        const logitsData = await this.backend.getData(tokenLogits);
        const token = argmax(logitsData);

        this.backend.dispose(tokenLogits);
        this.backend.dispose(predOut);

        if (token === this.blankId) {
          this.predNet.disposeState(newState);
          break;
        }

        tokens.push(token);
        this.predNet.disposeState(state);
        state = newState;
        lastToken = token;
        symbolsEmitted++;
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
  ): Promise<{ tokens: number[]; newState: PredictionState; newLastToken: number }> {
    const T = this.backend.getShape(encoderOutput)[1] as number;
    const D = this.backend.getShape(encoderOutput)[2] as number;
    const tokens: number[] = [];
    let currentState = state ?? this.predNet.initialState();
    let currentLastToken = lastToken;

    for (let t = 0; t < T; t++) {
      const encFrame = this.backend.slice(encoderOutput, [0, t, 0], [1, 1, D]);
      let symbolsEmitted = 0;

      while (symbolsEmitted < this.maxSymbolsPerStep) {
        const { output: predOut, newState } = this.predNet.step(currentLastToken, currentState);
        const { tokenLogits } = this.jointNet.forward(encFrame, predOut);
        const logitsData = await this.backend.getData(tokenLogits);
        const token = argmax(logitsData);

        this.backend.dispose(tokenLogits);
        this.backend.dispose(predOut);

        if (token === this.blankId) {
          this.predNet.disposeState(newState);
          break;
        }

        tokens.push(token);
        this.predNet.disposeState(currentState);
        currentState = newState;
        currentLastToken = token;
        symbolsEmitted++;
      }

      this.backend.dispose(encFrame);
    }

    return {
      tokens,
      newState: currentState,
      newLastToken: currentLastToken,
    };
  }

  createInitialState(): PredictionState {
    return this.predNet.initialState();
  }

  disposeState(state: PredictionState): void {
    this.predNet.disposeState(state);
  }
}
