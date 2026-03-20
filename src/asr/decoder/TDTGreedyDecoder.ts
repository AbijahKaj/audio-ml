import type { ComputeBackend, TensorHandle } from '../compute/index.js';
import type { PredictionNetwork, LSTMState } from './PredictionNetwork.js';
import type { TDTJointNetwork } from './TDTJointNetwork.js';

/**
 * TDT (Token-and-Duration Transducer) Greedy Decoder.
 *
 * Key difference from RNNT: the joint network predicts both a token AND a
 * duration — how many encoder frames to skip forward. This allows the decoder
 * to skip 2-4 frames at a time during silence/sustained sounds, achieving
 * 2-5× fewer decoder iterations than RNNT.
 *
 * Algorithm (Xu et al. ICML 2023):
 *   t = 0
 *   While t < T:
 *     While symbols emitted < maxSymbolsPerStep:
 *       1. Run prediction network with last token
 *       2. Run TDT joint network → (token_logits, duration_logits)
 *       3. token = argmax(token_logits), dur = durations[argmax(duration_logits)]
 *       4. If blank → advance t += max(1, dur), break inner loop
 *       5. If non-blank → emit token, advance t += dur
 *                         If dur > 0 → break inner loop (new encoder frame)
 */
export class TDTGreedyDecoder {
  private readonly blankId: number;
  private readonly maxSymbolsPerStep: number;
  private readonly durations: number[];

  constructor(
    private readonly backend: ComputeBackend,
    private readonly predNet: PredictionNetwork,
    private readonly jointNet: TDTJointNetwork,
    blankId = 0,
    durations: number[] = [0, 1, 2, 3, 4],
    maxSymbolsPerStep = 10,
  ) {
    this.blankId = blankId;
    this.durations = durations;
    this.maxSymbolsPerStep = maxSymbolsPerStep;
  }

  async decode(encoderOutput: TensorHandle): Promise<number[]> {
    const shape = this.backend.getShape(encoderOutput);
    const T = shape[1] as number;
    const dModel = shape[2] as number;
    const tokens: number[] = [];

    let state = this.predNet.initialState();
    let lastToken = this.blankId;
    let t = 0;

    while (t < T) {
      const encFrame = this.backend.slice(encoderOutput, [0, t, 0], [1, 1, dModel]);
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
          disposeState(this.backend, newState);
          t += Math.max(1, duration);
          break;
        }

        tokens.push(token);
        disposeState(this.backend, state);
        state = newState;
        lastToken = token;
        symbolsEmitted++;

        t += duration;
        if (duration > 0) break;  // new encoder frame needed
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
