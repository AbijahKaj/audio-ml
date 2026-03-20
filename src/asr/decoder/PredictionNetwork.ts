import type { ComputeBackend, TensorHandle } from '../compute/index.js';
import { ComputeScope } from '../compute/index.js';
import type { PredictionNetworkWeights } from '../model/WeightMapper.js';
import { Linear } from '../encoder/Linear.js';

export interface LSTMState {
  h: TensorHandle;  // [1, hidden]
  c: TensorHandle;  // [1, hidden]
}

/**
 * Prediction Network (shared by RNNT and TDT decoders).
 *
 * Architecture:
 *   token_id → embedding [embed_dim]
 *            → LSTM cell  [hidden_size]
 *            → Linear     [pred_hidden]
 *
 * This is a single-step implementation — processes one token at a time,
 * maintaining the LSTM state across steps (for greedy decoding).
 */
export class PredictionNetwork {
  private readonly embedding: TensorHandle;   // [vocab_size, embed_dim]
  private readonly lstmWeightIH: TensorHandle; // [4*hidden, embed_dim]
  private readonly lstmWeightHH: TensorHandle; // [4*hidden, hidden]
  private readonly lstmBiasIH: TensorHandle;   // [4*hidden]
  private readonly lstmBiasHH: TensorHandle;   // [4*hidden]
  private readonly outputProj: Linear;
  private readonly hiddenSize: number;

  constructor(
    private readonly backend: ComputeBackend,
    weights: PredictionNetworkWeights,
  ) {
    this.embedding = weights.embedding;
    this.lstmWeightIH = weights.lstmWeightIH;
    this.lstmWeightHH = weights.lstmWeightHH;
    this.lstmBiasIH = weights.lstmBiasIH;
    this.lstmBiasHH = weights.lstmBiasHH;
    this.outputProj = Linear.fromWeights(backend, weights.outputProj);

    // hidden size = bias length / 4
    const biasShape = backend.getShape(weights.lstmBiasIH);
    this.hiddenSize = (biasShape[0] as number) / 4;
  }

  initialState(): LSTMState {
    return {
      h: this.backend.zeros([1, this.hiddenSize]),
      c: this.backend.zeros([1, this.hiddenSize]),
    };
  }

  /**
   * One prediction network step.
   *
   * @param tokenId  Scalar token ID (last emitted token; blank=0 at start)
   * @param state    Previous LSTM state
   * @returns        { output: [1, pred_hidden], newState }
   */
  step(
    tokenId: number,
    state: LSTMState,
  ): { output: TensorHandle; newState: LSTMState } {
    const scope = new ComputeScope();

    // Embedding lookup: [1] index → [1, embed_dim]
    const idxTensor = scope.track(
      this.backend.tensor(new Int32Array([tokenId]), [1]),
    );
    const emb = scope.track(this.backend.gather(this.embedding, idxTensor, 0));
    const embReshaped = scope.track(this.backend.reshape(emb, [1, -1]));  // [1, embed_dim]

    // LSTM cell
    // gates = emb * W_ih^T + h * W_hh^T + b_ih + b_hh  → [1, 4*hidden]
    const wIHT = scope.track(this.backend.transpose(this.lstmWeightIH, [1, 0]));
    const wHHT = scope.track(this.backend.transpose(this.lstmWeightHH, [1, 0]));

    const ihOut = scope.track(this.backend.matmul(embReshaped, wIHT));
    const hhOut = scope.track(this.backend.matmul(state.h, wHHT));

    const gates = scope.track(
      this.backend.add(
        this.backend.add(
          scope.track(this.backend.add(ihOut, hhOut)),
          this.lstmBiasIH,
        ),
        this.lstmBiasHH,
      ),
    );

    // Split into i, f, g, o gates (each [1, hidden])
    const [iGate, fGate, gGate, oGate] = this.backend.split(gates, 4, 1);

    const i = scope.track(this.backend.sigmoid(iGate));
    const f = scope.track(this.backend.sigmoid(fGate));
    const g = scope.track(this.backend.tanh(gGate));
    const o = scope.track(this.backend.sigmoid(oGate));

    this.backend.dispose(iGate);
    this.backend.dispose(fGate);
    this.backend.dispose(gGate);
    this.backend.dispose(oGate);

    // c_new = f * c + i * g
    const fc = scope.track(this.backend.mul(f, state.c));
    const ig = scope.track(this.backend.mul(i, g));
    const cNew = this.backend.add(fc, ig);  // [1, hidden]

    // h_new = o * tanh(c_new)
    const tanhC = scope.track(this.backend.tanh(cNew));
    const hNew = this.backend.mul(o, tanhC);  // [1, hidden]

    // Output projection
    const output = this.outputProj.forward(hNew);

    scope.dispose(this.backend);
    return { output, newState: { h: hNew, c: cNew } };
  }
}
