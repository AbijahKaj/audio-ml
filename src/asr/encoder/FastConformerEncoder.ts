import type { ComputeBackend } from '../compute/Backend';
import type { TensorHandle } from '../compute/types';
import type { EncoderWeights } from '../model/WeightMapper';
import type { FastConformerConfig } from '../model/ModelConfig';
import { ConvSubsampling } from './ConvSubsampling';
import { ConformerBlock } from './ConformerBlock';

export interface StreamingEncoderState {
  cachedKV: Array<{ k: TensorHandle; v: TensorHandle }>;
  convStates: TensorHandle[];
}

export class FastConformerEncoder {
  private backend: ComputeBackend;
  private subsampling: ConvSubsampling;
  private blocks: ConformerBlock[];
  private finalNormWeight: TensorHandle | null;
  private finalNormBias: TensorHandle | null;
  private config: FastConformerConfig;

  constructor(backend: ComputeBackend, weights: EncoderWeights, config: FastConformerConfig) {
    this.backend = backend;
    this.config = config;
    this.subsampling = new ConvSubsampling(backend, weights.subsampling, config);
    this.blocks = weights.layers.map(
      layerWeights => new ConformerBlock(backend, layerWeights, config)
    );
    this.finalNormWeight = weights.finalNorm ? weights.finalNorm.weight : null;
    this.finalNormBias = weights.finalNorm ? weights.finalNorm.bias : null;
  }

  forward(melFeatures: TensorHandle): TensorHandle {
    return this.backend.tidy(() => {
      let x = this.subsampling.forward(melFeatures);
      for (const block of this.blocks) {
        x = block.forward(x);
      }
      if (this.finalNormWeight && this.finalNormBias) {
        x = this.backend.layerNorm(x, this.finalNormWeight, this.finalNormBias, 1e-5);
      }
      return x;
    });
  }

  forwardStreaming(
    melFeatures: TensorHandle,
    state: StreamingEncoderState | null,
  ): { output: TensorHandle; newState: StreamingEncoderState } {
    let x = this.subsampling.forward(melFeatures);

    const newKVs: Array<{ k: TensorHandle; v: TensorHandle }> = [];
    const newConvStates: TensorHandle[] = [];

    for (let i = 0; i < this.blocks.length; i++) {
      const cachedK = state ? state.cachedKV[i].k : null;
      const cachedV = state ? state.cachedKV[i].v : null;
      const convState = state ? state.convStates[i] : null;

      const result = this.blocks[i].forwardStreaming(x, cachedK, cachedV, convState);
      const prevX = x;
      x = result.output;
      this.backend.dispose(prevX);
      newKVs.push({ k: result.newK, v: result.newV });
      newConvStates.push(result.newConvState);
    }

    if (this.finalNormWeight && this.finalNormBias) {
      const preNorm = x;
      x = this.backend.layerNorm(x, this.finalNormWeight, this.finalNormBias, 1e-5);
      this.backend.dispose(preNorm);
    }

    return {
      output: x,
      newState: {
        cachedKV: newKVs,
        convStates: newConvStates,
      },
    };
  }

  createInitialState(): StreamingEncoderState {
    return {
      cachedKV: this.blocks.map(() => ({
        k: this.backend.zeros([1, this.config.numHeads, 0, this.config.dModel / this.config.numHeads]),
        v: this.backend.zeros([1, this.config.numHeads, 0, this.config.dModel / this.config.numHeads]),
      })),
      convStates: this.blocks.map(() =>
        this.backend.zeros([1, 0, this.config.dModel])
      ),
    };
  }

  get numLayers(): number {
    return this.blocks.length;
  }
}
