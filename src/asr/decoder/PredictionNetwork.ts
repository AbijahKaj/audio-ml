import type { ComputeBackend } from '../compute/Backend';
import type { TensorHandle } from '../compute/types';
import type { PredictionNetworkWeights } from '../model/WeightMapper';
import { Linear } from '../encoder/Linear';

export interface PredictionState {
  h: TensorHandle[];
  c: TensorHandle[];
}

/**
 * LSTM-based prediction network for both RNNT and TDT decoders.
 * Conditions on previously emitted tokens.
 */
export class PredictionNetwork {
  private backend: ComputeBackend;
  private embedding: TensorHandle;
  private lstmWeightsIH: TensorHandle[];
  private lstmWeightsHH: TensorHandle[];
  private lstmBiasIH: TensorHandle[];
  private lstmBiasHH: TensorHandle[];
  private outputProj: Linear;
  private hiddenSize: number;
  private numLayers: number;

  constructor(backend: ComputeBackend, weights: PredictionNetworkWeights, hiddenSize: number) {
    this.backend = backend;
    this.embedding = weights.embedding;
    this.lstmWeightsIH = weights.lstmWeightsIH;
    this.lstmWeightsHH = weights.lstmWeightsHH;
    this.lstmBiasIH = weights.lstmBiasIH;
    this.lstmBiasHH = weights.lstmBiasHH;
    this.outputProj = new Linear(backend, weights.outputProj);
    this.hiddenSize = hiddenSize;
    this.numLayers = weights.lstmWeightsIH.length;
  }

  step(
    tokenId: number,
    state: PredictionState,
  ): { output: TensorHandle; newState: PredictionState } {
    // Embed the token
    const idxTensor = this.backend.tensor(new Int32Array([tokenId]), [1]);
    let x = this.backend.gather(this.embedding, idxTensor, 0); // [1, embed_dim]
    this.backend.dispose(idxTensor);

    const newH: TensorHandle[] = [];
    const newC: TensorHandle[] = [];

    // Run through LSTM layers
    for (let l = 0; l < this.numLayers; l++) {
      const result = this.lstmCell(x, state.h[l], state.c[l], l);
      x = result.h;
      newH.push(result.h);
      newC.push(result.c);
    }

    // Output projection
    const output = this.outputProj.forward(x);

    return {
      output,
      newState: { h: newH, c: newC },
    };
  }

  initialState(): PredictionState {
    return {
      h: Array.from({ length: this.numLayers }, () =>
        this.backend.zeros([1, this.hiddenSize])
      ),
      c: Array.from({ length: this.numLayers }, () =>
        this.backend.zeros([1, this.hiddenSize])
      ),
    };
  }

  disposeState(state: PredictionState): void {
    for (const h of state.h) this.backend.dispose(h);
    for (const c of state.c) this.backend.dispose(c);
  }

  private lstmCell(
    input: TensorHandle,
    prevH: TensorHandle,
    prevC: TensorHandle,
    layer: number,
  ): { h: TensorHandle; c: TensorHandle } {
    return this.backend.tidy(() => {
      // gates = input * W_ih^T + h * W_hh^T + b_ih + b_hh
      const wIH_T = this.backend.transpose(this.lstmWeightsIH[layer], [1, 0]);
      const wHH_T = this.backend.transpose(this.lstmWeightsHH[layer], [1, 0]);

      const inputGates = this.backend.matmul(input, wIH_T);
      const hiddenGates = this.backend.matmul(prevH, wHH_T);
      let gates = this.backend.add(inputGates, hiddenGates);
      gates = this.backend.add(gates, this.lstmBiasIH[layer]);
      gates = this.backend.add(gates, this.lstmBiasHH[layer]);

      // Split into 4 gate chunks: [i, f, g, o] each of size hidden_size
      const gateParts = this.backend.split(gates, 4, -1);
      const i = this.backend.sigmoid(gateParts[0]);
      const f = this.backend.sigmoid(gateParts[1]);
      const g = this.backend.tanh(gateParts[2]);
      const o = this.backend.sigmoid(gateParts[3]);

      // c_new = f * c_prev + i * g
      const c = this.backend.add(
        this.backend.mul(f, prevC),
        this.backend.mul(i, g)
      );

      // h_new = o * tanh(c_new)
      const h = this.backend.mul(o, this.backend.tanh(c));

      return { h, c };
    });
  }
}
