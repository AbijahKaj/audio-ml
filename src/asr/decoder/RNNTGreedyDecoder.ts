import type { ComputeBackend, TensorHandle } from '../compute/index.js';
import type { PredictionNetwork, LSTMState } from './PredictionNetwork.js';
import type { RNNTJointNetwork } from './RNNTJointNetwork.js';

/**
 * RNNT Greedy Decoder.
 *
 * Algorithm (Graves 2012):
 *   For each encoder frame t = 0..T-1:
 *     While symbols emitted < maxSymbolsPerStep:
 *       1. Run prediction network with last token
 *       2. Run joint network
 *       3. Argmax over logits
 *       4. If blank → advance to next frame
 *       5. If non-blank → emit token, stay at same frame
 *
 * Reference: Graves (2012) — "Sequence Transduction with Recurrent Neural Networks"
 */
export class RNNTGreedyDecoder {
  private readonly blankId: number;
  private readonly maxSymbolsPerStep: number;

  constructor(
    private readonly backend: ComputeBackend,
    private readonly predNet: PredictionNetwork,
    private readonly jointNet: RNNTJointNetwork,
    blankId = 0,
    maxSymbolsPerStep = 10,
  ) {
    this.blankId = blankId;
    this.maxSymbolsPerStep = maxSymbolsPerStep;
  }

  async decode(encoderOutput: TensorHandle): Promise<number[]> {
    const shape = this.backend.getShape(encoderOutput);
    const T = shape[1] as number;
    const tokens: number[] = [];

    let state = this.predNet.initialState();
    let lastToken = this.blankId;

    for (let t = 0; t < T; t++) {
      const encFrame = this.backend.slice(
        encoderOutput, [0, t, 0], [1, 1, shape[2] as number],
      );

      let symbolsEmitted = 0;

      while (symbolsEmitted < this.maxSymbolsPerStep) {
        const { output: predOut, newState } = this.predNet.step(lastToken, state);
        const { tokenLogits } = this.jointNet.forward(encFrame, predOut);

        const logitsData = await this.backend.getData(tokenLogits);
        const token = argmax(logitsData);

        this.backend.dispose(tokenLogits);
        this.backend.dispose(predOut);

        if (token === this.blankId) {
          disposeState(this.backend, newState);
          break;
        }

        tokens.push(token);
        disposeState(this.backend, state);
        state = newState;
        lastToken = token;
        symbolsEmitted++;
      }

      this.backend.dispose(encFrame);
    }

    disposeState(this.backend, state);
    return tokens;
  }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function argmax(data: Float32Array): number {
  let best = 0;
  let bestVal = data[0];
  for (let i = 1; i < data.length; i++) {
    if (data[i] > bestVal) {
      bestVal = data[i];
      best = i;
    }
  }
  return best;
}

function disposeState(backend: ComputeBackend, state: LSTMState): void {
  backend.dispose(state.h);
  backend.dispose(state.c);
}
