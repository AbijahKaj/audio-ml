import type { TensorHandle } from '../compute/types';

export interface LayerNormParams {
  weight: TensorHandle;
  bias: TensorHandle;
}

export interface LinearParams {
  weight: TensorHandle;
  bias: TensorHandle | null;
}

export interface SubsamplingConvLayer {
  weight: TensorHandle;
  bias: TensorHandle | null;
}

export interface SubsamplingWeights {
  convLayers: SubsamplingConvLayer[];
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
  norm: LayerNormParams;
  pointwise1: LinearParams;
  depthwiseWeight: TensorHandle;
  depthwiseBias: TensorHandle | null;
  batchNorm: {
    mean: TensorHandle;
    variance: TensorHandle;
    scale: TensorHandle;
    offset: TensorHandle;
  };
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
