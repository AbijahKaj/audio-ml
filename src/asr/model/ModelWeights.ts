import type { TensorHandle } from '../compute/types';

export interface LayerNormWeights {
  weight: TensorHandle;
  bias: TensorHandle;
}

export interface LinearWeights {
  weight: TensorHandle;
  bias?: TensorHandle;
}

export interface BatchNormWeights {
  mean?: TensorHandle;
  variance?: TensorHandle;
  scale: TensorHandle;
  offset: TensorHandle;
}

export interface ConvSubsamplingWeights {
  conv1: LinearWeights;
  conv2: LinearWeights;
  conv3?: LinearWeights;
  out: LinearWeights;
}

export interface FeedForwardWeights {
  norm: LayerNormWeights;
  linear1: LinearWeights;
  linear2: LinearWeights;
}

export interface AttentionWeights {
  norm: LayerNormWeights;
  q: LinearWeights;
  k: LinearWeights;
  v: LinearWeights;
  out: LinearWeights;
  posBiasU?: TensorHandle;
  posBiasV?: TensorHandle;
}

export interface ConvModuleWeights {
  norm: LayerNormWeights;
  pointwise1: LinearWeights;
  depthwiseKernel: TensorHandle;
  batchNorm: BatchNormWeights;
  pointwise2: LinearWeights;
}

export interface ConformerLayerWeights {
  ffn1: FeedForwardWeights;
  attn: AttentionWeights;
  conv: ConvModuleWeights;
  ffn2: FeedForwardWeights;
  finalNorm: LayerNormWeights;
}

export interface EncoderWeights {
  subsampling: ConvSubsamplingWeights;
  layers: ConformerLayerWeights[];
}

export interface PredictionWeights {
  embedding: TensorHandle;
  lstmWeightIH: TensorHandle;
  lstmWeightHH: TensorHandle;
  lstmBiasIH: TensorHandle;
  lstmBiasHH: TensorHandle;
  outputProj?: LinearWeights;
}

export interface JointWeights {
  encoderProj: LinearWeights;
  predictionProj: LinearWeights;
  hiddenProj?: LinearWeights;
  tokenProj: LinearWeights;
  durationProj?: LinearWeights;
}

export interface DecoderWeights {
  prediction: PredictionWeights;
  joint: JointWeights;
}

export interface ModelWeights {
  encoder: EncoderWeights;
  decoder: DecoderWeights;
  consumedKeys: string[];
  unusedKeys: string[];
  raw: Map<string, TensorHandle>;
}
