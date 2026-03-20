import type { TensorHandle } from '../compute/types';
import type { FastConformerConfig } from './ModelConfig';

export interface LayerNormWeights {
  weight: TensorHandle;
  bias: TensorHandle;
}

export interface LinearWeights {
  weight: TensorHandle;
  bias: TensorHandle | null;
}

export interface FeedForwardWeights {
  norm: LayerNormWeights;
  linear1: LinearWeights;
  linear2: LinearWeights;
}

export interface AttentionWeights {
  norm: LayerNormWeights;
  queryProj: LinearWeights;
  keyProj: LinearWeights;
  valueProj: LinearWeights;
  outProj: LinearWeights;
  posBias: TensorHandle | null;
  posProj: LinearWeights | null;
}

export interface ConvModuleWeights {
  norm: LayerNormWeights;
  pointwise1: LinearWeights;
  depthwiseWeight: TensorHandle;
  depthwiseBias: TensorHandle | null;
  batchNorm: {
    weight: TensorHandle;
    bias: TensorHandle;
    runningMean: TensorHandle;
    runningVar: TensorHandle;
  };
  pointwise2: LinearWeights;
}

export interface ConformerBlockWeights {
  ffn1: FeedForwardWeights;
  attn: AttentionWeights;
  conv: ConvModuleWeights;
  ffn2: FeedForwardWeights;
  finalNorm: LayerNormWeights;
}

export interface SubsamplingWeights {
  conv1Weight: TensorHandle;
  conv1Bias: TensorHandle;
  conv2Weight: TensorHandle;
  conv2Bias: TensorHandle;
  outWeight: TensorHandle;
  outBias: TensorHandle;
}

export interface EncoderWeights {
  subsampling: SubsamplingWeights;
  layers: ConformerBlockWeights[];
  finalNorm: LayerNormWeights | null;
}

export interface PredictionNetworkWeights {
  embedding: TensorHandle;
  lstmWeightsIH: TensorHandle[];
  lstmWeightsHH: TensorHandle[];
  lstmBiasIH: TensorHandle[];
  lstmBiasHH: TensorHandle[];
  outputProj: LinearWeights;
}

export interface JointNetworkWeights {
  encoderProj: LinearWeights;
  predictionProj: LinearWeights;
  outputProj: LinearWeights;
  durationProj?: LinearWeights;
}

export interface DecoderWeights {
  prediction: PredictionNetworkWeights;
  joint: JointNetworkWeights;
}

export interface ModelWeights {
  encoder: EncoderWeights;
  decoder: DecoderWeights;
}

export function mapWeights(
  weights: Map<string, TensorHandle>,
  config: FastConformerConfig
): ModelWeights {
  const consumed = new Set<string>();

  function get(name: string): TensorHandle {
    if (!weights.has(name)) throw new Error(`Missing weight: ${name}`);
    consumed.add(name);
    return weights.get(name)!;
  }

  function tryGet(name: string): TensorHandle | null {
    if (!weights.has(name)) return null;
    consumed.add(name);
    return weights.get(name)!;
  }

  function getLinear(prefix: string): LinearWeights {
    return {
      weight: get(`${prefix}.weight`),
      bias: tryGet(`${prefix}.bias`),
    };
  }

  function getLayerNorm(prefix: string): LayerNormWeights {
    return {
      weight: get(`${prefix}.weight`),
      bias: get(`${prefix}.bias`),
    };
  }

  const subsampling: SubsamplingWeights = {
    conv1Weight: get('encoder.pre_encode.conv.0.weight'),
    conv1Bias: get('encoder.pre_encode.conv.0.bias'),
    conv2Weight: get('encoder.pre_encode.conv.2.weight'),
    conv2Bias: get('encoder.pre_encode.conv.2.bias'),
    outWeight: get('encoder.pre_encode.out.0.weight'),
    outBias: get('encoder.pre_encode.out.0.bias'),
  };

  const layers: ConformerBlockWeights[] = [];
  for (let i = 0; i < config.encoderLayers; i++) {
    const prefix = `encoder.layers.${i}`;

    const ffn1: FeedForwardWeights = {
      norm: getLayerNorm(`${prefix}.norm_feed_forward1`),
      linear1: getLinear(`${prefix}.fc1`),
      linear2: getLinear(`${prefix}.fc2`),
    };

    const attn: AttentionWeights = {
      norm: getLayerNorm(`${prefix}.norm_self_att`),
      queryProj: getLinear(`${prefix}.self_attn.linear_q`),
      keyProj: getLinear(`${prefix}.self_attn.linear_k`),
      valueProj: getLinear(`${prefix}.self_attn.linear_v`),
      outProj: getLinear(`${prefix}.self_attn.linear_out`),
      posBias: tryGet(`${prefix}.self_attn.pos_bias_u`) ?? tryGet(`${prefix}.self_attn.pos_bias`),
      posProj: tryGet(`${prefix}.self_attn.linear_pos.weight`)
        ? getLinear(`${prefix}.self_attn.linear_pos`)
        : null,
    };

    const conv: ConvModuleWeights = {
      norm: getLayerNorm(`${prefix}.norm_conv`),
      pointwise1: getLinear(`${prefix}.conv_module.pointwise_conv1`),
      depthwiseWeight: get(`${prefix}.conv_module.depthwise_conv.weight`),
      depthwiseBias: tryGet(`${prefix}.conv_module.depthwise_conv.bias`),
      batchNorm: {
        weight: get(`${prefix}.conv_module.batch_norm.weight`),
        bias: get(`${prefix}.conv_module.batch_norm.bias`),
        runningMean: get(`${prefix}.conv_module.batch_norm.running_mean`),
        runningVar: get(`${prefix}.conv_module.batch_norm.running_var`),
      },
      pointwise2: getLinear(`${prefix}.conv_module.pointwise_conv2`),
    };

    const ffn2: FeedForwardWeights = {
      norm: getLayerNorm(`${prefix}.norm_feed_forward2`),
      linear1: getLinear(`${prefix}.fc3`),
      linear2: getLinear(`${prefix}.fc4`),
    };

    const finalNorm = getLayerNorm(`${prefix}.norm_out`);

    layers.push({ ffn1, attn, conv, ffn2, finalNorm });
  }

  const encoderFinalNorm: LayerNormWeights | null =
    tryGet('encoder.norm.weight')
      ? getLayerNorm('encoder.norm')
      : null;

  const predictionLayers = config.predNumLayers;
  const lstmWeightsIH: TensorHandle[] = [];
  const lstmWeightsHH: TensorHandle[] = [];
  const lstmBiasIH: TensorHandle[] = [];
  const lstmBiasHH: TensorHandle[] = [];
  for (let l = 0; l < predictionLayers; l++) {
    lstmWeightsIH.push(get(`decoder.prediction.lstm.weight_ih_l${l}`));
    lstmWeightsHH.push(get(`decoder.prediction.lstm.weight_hh_l${l}`));
    lstmBiasIH.push(get(`decoder.prediction.lstm.bias_ih_l${l}`));
    lstmBiasHH.push(get(`decoder.prediction.lstm.bias_hh_l${l}`));
  }

  const prediction: PredictionNetworkWeights = {
    embedding: get('decoder.prediction.embed.weight'),
    lstmWeightsIH,
    lstmWeightsHH,
    lstmBiasIH,
    lstmBiasHH,
    outputProj: getLinear('decoder.prediction.dec_rnn_layers.0'),
  };

  const joint: JointNetworkWeights = {
    encoderProj: getLinear('joint.joint_net.0'),
    predictionProj: getLinear('joint.pred_net.0'),
    outputProj: getLinear('joint.joint_net.2'),
  };

  if (config.decoderType === 'tdt') {
    joint.durationProj = tryGet('joint.duration_net.weight')
      ? getLinear('joint.duration_net')
      : undefined;
  }

  const unconsumed = [...weights.keys()].filter(k => !consumed.has(k) && k !== '__metadata__');
  if (unconsumed.length > 0) {
    console.warn('Unconsumed weights:', unconsumed);
  }

  return {
    encoder: {
      subsampling,
      layers,
      finalNorm: encoderFinalNorm,
    },
    decoder: {
      prediction,
      joint,
    },
  };
}
