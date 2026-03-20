import type { TensorHandle } from '../compute/types';

export interface LayerNormParams {
  weight: TensorHandle;
  bias: TensorHandle;
}

export interface LinearParams {
  weight: TensorHandle;
  bias: TensorHandle | null;
}

/** NeMo ConvSubsampling: striding uses stacked 3×3 stride-2 convs; dw_striding uses conv + (depthwise+pointwise)×k. */
export type SubsamplingLayerSpec =
  | { kind: 'conv2d_s2'; weight: TensorHandle; bias: TensorHandle | null }
  | { kind: 'depthwise_s2'; weight: TensorHandle; bias: TensorHandle | null }
  | { kind: 'pointwise1x1'; weight: TensorHandle; bias: TensorHandle | null };

export interface SubsamplingWeights {
  layers: SubsamplingLayerSpec[];
  out: LinearParams;
}

export interface SelfAttentionWeights {
  linearQ: LinearParams;
  linearK: LinearParams;
  linearV: LinearParams;
  linearOut: LinearParams;
  linearPos: LinearParams;
  posBiasU: TensorHandle;
  posBiasV: TensorHandle;
}

export interface ConvModuleWeights {
  /** After depthwise: NeMo uses `batch_norm.*` for LayerNorm or BatchNorm1d. */
  afterDepthwise: LayerNormParams;
  useBatchNormStats: boolean;
  bnRunningMean: TensorHandle | null;
  bnRunningVar: TensorHandle | null;
  pointwise1: LinearParams;
  depthwiseWeight: TensorHandle;
  depthwiseBias: TensorHandle | null;
  pointwise2: LinearParams;
}

export interface ConformerLayerWeights {
  normFeedForward1: LayerNormParams;
  feedForward1: { linear1: LinearParams; linear2: LinearParams };
  normSelfAtt: LayerNormParams;
  selfAtt: SelfAttentionWeights;
  normConv: LayerNormParams;
  conv: ConvModuleWeights;
  normFeedForward2: LayerNormParams;
  feedForward2: { linear1: LinearParams; linear2: LinearParams };
  normOut: LayerNormParams;
}

export interface EncoderWeights {
  subsampling: SubsamplingWeights;
  layers: ConformerLayerWeights[];
}

export interface PredictionWeights {
  embedding: TensorHandle;
  lstm: {
    weightIh: TensorHandle;
    weightHh: TensorHandle;
    biasIh: TensorHandle;
    biasHh: TensorHandle;
  };
  numLayers: number;
}

export interface JointWeights {
  enc: LinearParams;
  pred: LinearParams;
  jointNetLinear: LinearParams;
  /** If TDT: separate duration logits layer (optional). */
  durationLinear?: LinearParams;
}

export interface ModelWeights {
  encoder: EncoderWeights;
  decoder: {
    prediction: PredictionWeights;
    joint: JointWeights;
  };
}
