import { ComputeScope } from '../compute/Scope.js';
import type { ComputeBackend } from '../compute/Backend.js';
import type { TensorHandle } from '../compute/types.js';
import type { ConformerLayerWeights } from '../model/WeightMapper.js';
import { ConvModule } from './ConvModule.js';
import { FeedForward } from './FeedForward.js';
import { MultiHeadAttention } from './MultiHeadAttention.js';

export class ConformerBlock {
  private readonly ffn1: FeedForward;
  private readonly attention: MultiHeadAttention;
  private readonly conv: ConvModule;
  private readonly ffn2: FeedForward;

  constructor(
    private readonly backend: ComputeBackend,
    private readonly weights: ConformerLayerWeights,
    numHeads: number,
    convKernelSize: number,
  ) {
    this.ffn1 = new FeedForward(backend, weights.ffn1);
    this.attention = new MultiHeadAttention(backend, weights.attention, numHeads);
    this.conv = new ConvModule(backend, weights.conv, Math.floor(convKernelSize / 2));
    this.ffn2 = new FeedForward(backend, weights.ffn2);
  }

  forward(input: TensorHandle): TensorHandle {
    const scope = new ComputeScope();
    let state = input;
    state = this.addResidual(state, scope.track(this.backend.scale(this.ffn1.forward(state), 0.5)));
    state = this.addResidual(state, this.attention.forward(state));
    state = this.addResidual(state, this.conv.forward(state));
    state = this.addResidual(state, scope.track(this.backend.scale(this.ffn2.forward(state), 0.5)));

    if (this.weights.finalNorm.weight && this.weights.finalNorm.bias) {
      const normalized = this.backend.layerNorm(state, this.weights.finalNorm.weight, this.weights.finalNorm.bias, 1e-5);
      if (normalized !== state) {
        this.backend.dispose(state);
      }
      state = normalized;
    }

    scope.dispose(this.backend);
    return state;
  }

  private addResidual(input: TensorHandle, delta: TensorHandle): TensorHandle {
    if (delta === input) {
      return input;
    }

    const summed = this.backend.add(input, delta);
    this.backend.dispose(delta);
    if (summed !== input) {
      this.backend.dispose(input);
    }
    return summed;
  }
}
