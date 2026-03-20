import type { TensorHandle } from '../compute/index.js';
import type { FastConformerConfig } from './ModelConfig.js';

/**
 * Structured weight tree for the FastConformer model.
 * Mirrors the NeMo model's module hierarchy.
 */
export interface LayerNormWeights {
  weight: TensorHandle;
  bias: TensorHandle;
}

export interface LinearWeights {
  weight: TensorHandle;
  bias: TensorHandle;
}

export interface FeedForwardWeights {
  norm: LayerNormWeights;
  fc1: LinearWeights;
  fc2: LinearWeights;
}

export interface AttentionWeights {
  norm: LayerNormWeights;
  queryProj: LinearWeights;
  keyProj: LinearWeights;
  valueProj: LinearWeights;
  outProj: LinearWeights;
  posProj: LinearWeights;
  posU: TensorHandle;  // learnable bias u for relative pos encoding
  posV: TensorHandle;  // learnable bias v for relative pos encoding
}

export interface ConvModuleWeights {
  norm: LayerNormWeights;
  pointwiseConv1: { weight: TensorHandle; bias: TensorHandle };
  depthwiseConv: { weight: TensorHandle };
  batchNorm: { weight: TensorHandle; bias: TensorHandle; mean: TensorHandle; var: TensorHandle };
  pointwiseConv2: { weight: TensorHandle; bias: TensorHandle };
}

export interface ConformerBlockWeights {
  ffn1: FeedForwardWeights;
  attn: AttentionWeights;
  conv: ConvModuleWeights;
  ffn2: FeedForwardWeights;
  finalNorm: LayerNormWeights;
}

export interface SubsamplingWeights {
  conv1: { weight: TensorHandle; bias: TensorHandle };
  conv2: { weight: TensorHandle; bias: TensorHandle };
  out: { weight: TensorHandle; bias: TensorHandle };
}

export interface EncoderWeights {
  subsampling: SubsamplingWeights;
  layers: ConformerBlockWeights[];
}

export interface PredictionNetworkWeights {
  embedding: TensorHandle;             // [vocab_size, embed_dim]
  lstmWeightIH: TensorHandle;          // [4*hidden, embed_dim]
  lstmWeightHH: TensorHandle;          // [4*hidden, hidden]
  lstmBiasIH: TensorHandle;            // [4*hidden]
  lstmBiasHH: TensorHandle;            // [4*hidden]
  outputProj: LinearWeights;           // hidden → pred_hidden
}

export interface JointNetworkWeights {
  encoderProj: LinearWeights;
  predictionProj: LinearWeights;
  tokenProj: LinearWeights;
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

/**
 * Map NeMo SafeTensors parameter names → structured ModelWeights.
 *
 * NeMo FastConformer naming conventions (discovered via model.state_dict().keys()):
 *   encoder.pre_encode.conv.0.weight          — subsampling conv1
 *   encoder.pre_encode.out.0.weight            — subsampling linear
 *   encoder.layers.{i}.norm_feed_forward1.*    — FFN1 layer norm
 *   encoder.layers.{i}.feed_forward1.linear1.* — FFN1 linear layers
 *   encoder.layers.{i}.self_attn.*             — attention
 *   encoder.layers.{i}.conv_module.*           — conv module
 *   encoder.layers.{i}.feed_forward2.*         — FFN2
 *   encoder.layers.{i}.norm_out.*              — final layer norm
 *   decoder.prediction.embed.weight            — token embedding
 *   decoder.prediction.dec_rnn.lstm.weight_ih_l0
 *   decoder.joint.enc.weight                   — joint encoder proj
 */
export function mapWeights(
  weights: Map<string, TensorHandle>,
  config: FastConformerConfig,
): ModelWeights {
  const consumed = new Set<string>();

  function get(name: string): TensorHandle {
    if (!weights.has(name)) {
      throw new Error(
        `Missing weight: "${name}". ` +
        `Run tools/export_nemo_to_safetensors.py and check the _weight_keys.txt file.`,
      );
    }
    consumed.add(name);
    return weights.get(name)!;
  }

  function tryGet(name: string, fallback: string): TensorHandle {
    if (weights.has(name)) {
      consumed.add(name);
      return weights.get(name)!;
    }
    return get(fallback);
  }

  function getLayerNorm(prefix: string): LayerNormWeights {
    return {
      weight: get(`${prefix}.weight`),
      bias: get(`${prefix}.bias`),
    };
  }

  function getLinear(prefix: string): LinearWeights {
    return {
      weight: get(`${prefix}.weight`),
      bias: get(`${prefix}.bias`),
    };
  }

  function getFeedForward(i: number, which: 1 | 2): FeedForwardWeights {
    const normKey = `encoder.layers.${i}.norm_feed_forward${which}`;
    const ffKey = `encoder.layers.${i}.feed_forward${which}`;
    return {
      norm: getLayerNorm(normKey),
      fc1: getLinear(`${ffKey}.linear1`),
      fc2: getLinear(`${ffKey}.linear2`),
    };
  }

  function getAttention(i: number): AttentionWeights {
    const base = `encoder.layers.${i}.self_attn`;
    const norm = getLayerNorm(`encoder.layers.${i}.norm_self_att`);
    return {
      norm,
      queryProj: getLinear(`${base}.linear_q`),
      keyProj: getLinear(`${base}.linear_k`),
      valueProj: getLinear(`${base}.linear_v`),
      outProj: getLinear(`${base}.linear_out`),
      posProj: getLinear(`${base}.linear_pos`),
      posU: get(`${base}.pos_bias_u`),
      posV: get(`${base}.pos_bias_v`),
    };
  }

  function getConvModule(i: number): ConvModuleWeights {
    const base = `encoder.layers.${i}.conv_module`;
    return {
      norm: getLayerNorm(`encoder.layers.${i}.norm_conv_module`),
      pointwiseConv1: {
        weight: get(`${base}.pointwise_conv1.weight`),
        bias: get(`${base}.pointwise_conv1.bias`),
      },
      depthwiseConv: {
        weight: get(`${base}.depthwise_conv.weight`),
      },
      batchNorm: {
        weight: get(`${base}.batch_norm.weight`),
        bias: get(`${base}.batch_norm.bias`),
        mean: get(`${base}.batch_norm.running_mean`),
        var: get(`${base}.batch_norm.running_var`),
      },
      pointwiseConv2: {
        weight: get(`${base}.pointwise_conv2.weight`),
        bias: get(`${base}.pointwise_conv2.bias`),
      },
    };
  }

  const subsampling: SubsamplingWeights = {
    conv1: {
      weight: tryGet('encoder.pre_encode.conv.0.weight', 'encoder.pre_encode.conv.0.weight'),
      bias: tryGet('encoder.pre_encode.conv.0.bias', 'encoder.pre_encode.conv.0.bias'),
    },
    conv2: {
      weight: get('encoder.pre_encode.conv.2.weight'),
      bias: get('encoder.pre_encode.conv.2.bias'),
    },
    out: {
      weight: get('encoder.pre_encode.out.0.weight'),
      bias: get('encoder.pre_encode.out.0.bias'),
    },
  };

  const layers: ConformerBlockWeights[] = Array.from(
    { length: config.encoderLayers },
    (_, i) => ({
      ffn1: getFeedForward(i, 1),
      attn: getAttention(i),
      conv: getConvModule(i),
      ffn2: getFeedForward(i, 2),
      finalNorm: getLayerNorm(`encoder.layers.${i}.norm_out`),
    }),
  );

  // Prediction network
  const predBase = 'decoder.prediction';
  const prediction: PredictionNetworkWeights = {
    embedding: get(`${predBase}.embed.weight`),
    lstmWeightIH: get(`${predBase}.dec_rnn.lstm.weight_ih_l0`),
    lstmWeightHH: get(`${predBase}.dec_rnn.lstm.weight_hh_l0`),
    lstmBiasIH: get(`${predBase}.dec_rnn.lstm.bias_ih_l0`),
    lstmBiasHH: get(`${predBase}.dec_rnn.lstm.bias_hh_l0`),
    outputProj: getLinear(`${predBase}.predict_net`),
  };

  // Joint network
  const jointBase = 'decoder.joint';
  const joint: JointNetworkWeights = {
    encoderProj: getLinear(`${jointBase}.enc`),
    predictionProj: getLinear(`${jointBase}.pred`),
    tokenProj: getLinear(`${jointBase}.joint_net`),
  };

  if (config.decoderType === 'tdt') {
    joint.durationProj = getLinear(`${jointBase}.duration_head`);
  }

  // Warn about unconsumed weights (helps catch name mismatches)
  const unconsumed = [...weights.keys()].filter(k => !consumed.has(k));
  if (unconsumed.length > 0) {
    console.warn(
      `WeightMapper: ${unconsumed.length} unconsumed weights. ` +
      `First 10: ${unconsumed.slice(0, 10).join(', ')}`,
    );
  }

  return {
    encoder: { subsampling, layers },
    decoder: { prediction, joint },
  };
}
