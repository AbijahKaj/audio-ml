import type { TensorHandle } from '../compute/types';

export interface LinearWeights {
  weight: TensorHandle;
  bias: TensorHandle;
}

export interface LayerNormWeights {
  weight: TensorHandle;
  bias: TensorHandle;
}

export interface BatchNormWeights {
  runningMean: TensorHandle;
  runningVar: TensorHandle;
  weight: TensorHandle;
  bias: TensorHandle;
}

export interface SubsamplingWeights {
  conv0: { weight: TensorHandle; bias: TensorHandle };
  conv1: { weight: TensorHandle; bias: TensorHandle };
  conv2: { weight: TensorHandle; bias: TensorHandle };
  out: LinearWeights;
}

export interface RelPosAttnWeights {
  linearQ: LinearWeights;
  linearK: LinearWeights;
  linearV: LinearWeights;
  linearOut: LinearWeights;
  linearPos: LinearWeights;
  posBiasU: TensorHandle;
  posBiasV: TensorHandle;
}

export interface ConformerConvWeights {
  pointwise1: LinearWeights;
  depthwise: TensorHandle;
  depthwiseBias: TensorHandle;
  batchNorm: BatchNormWeights;
  pointwise2: LinearWeights;
}

export interface ConformerLayerWeights {
  normFeedForward1: LayerNormWeights;
  feedForward1: { w1: LinearWeights; w2: LinearWeights };
  normSelfAtt: LayerNormWeights;
  selfAtt: RelPosAttnWeights;
  normConv: LayerNormWeights;
  conv: ConformerConvWeights;
  normFeedForward2: LayerNormWeights;
  feedForward2: { w1: LinearWeights; w2: LinearWeights };
  normOut: LayerNormWeights;
}

export interface EncoderWeights {
  subsampling: SubsamplingWeights;
  layers: ConformerLayerWeights[];
}

export interface PredictionWeights {
  embedding: TensorHandle;
  lstmWih: TensorHandle;
  lstmWhh: TensorHandle;
  lstmBias: TensorHandle;
  /** Optional projection after LSTM before joint (some checkpoints fold into joint) */
  proj?: LinearWeights;
}

export interface RNNTJointWeights {
  encoderProj: LinearWeights;
  predProj: LinearWeights;
  output: LinearWeights;
}

export interface TDTJointWeights extends RNNTJointWeights {
  duration: LinearWeights;
}

export interface DecoderWeights {
  prediction: PredictionWeights;
  joint: RNNTJointWeights | TDTJointWeights;
}

export interface ModelWeights {
  encoder: EncoderWeights;
  decoder: DecoderWeights;
}