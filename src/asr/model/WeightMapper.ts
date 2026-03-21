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
  posBiasU: TensorHandle;
  posBiasV: TensorHandle;
  posProj: LinearWeights | null;
}

export interface ConvModuleWeights {
  norm: LayerNormWeights;
  pointwise1Weight: TensorHandle;
  pointwise1Bias: TensorHandle | null;
  depthwiseWeight: TensorHandle;
  depthwiseBias: TensorHandle | null;
  batchNorm: {
    weight: TensorHandle;
    bias: TensorHandle;
    runningMean: TensorHandle;
    runningVar: TensorHandle;
  };
  pointwise2Weight: TensorHandle;
  pointwise2Bias: TensorHandle | null;
}

export interface ConformerBlockWeights {
  ffn1: FeedForwardWeights;
  attn: AttentionWeights;
  conv: ConvModuleWeights;
  ffn2: FeedForwardWeights;
  finalNorm: LayerNormWeights;
}

export interface SubsamplingWeights {
  allConvWeights: TensorHandle[];
  allConvBiases: TensorHandle[];
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
  outputProj: LinearWeights | null;
}

export interface JointNetworkWeights {
  encoderProj: LinearWeights;
  predictionProj: LinearWeights;
  outputProj: LinearWeights;
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
  config: FastConformerConfig,
  backend?: import('../compute/Backend').ComputeBackend,
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

  function tryGetLinear(prefix: string): LinearWeights | null {
    const w = tryGet(`${prefix}.weight`);
    if (!w) return null;
    return {
      weight: w,
      bias: tryGet(`${prefix}.bias`),
    };
  }

  function getLayerNorm(prefix: string): LayerNormWeights {
    return {
      weight: get(`${prefix}.weight`),
      bias: get(`${prefix}.bias`),
    };
  }

  // === Subsampling ===
  // FastConformer dw_striding: multiple conv layers with varying indices
  const allConvWeights: TensorHandle[] = [];
  const allConvBiases: TensorHandle[] = [];
  for (let idx = 0; idx < 10; idx++) {
    const w = tryGet(`encoder.pre_encode.conv.${idx}.weight`);
    if (w) {
      allConvWeights.push(w);
      const b = tryGet(`encoder.pre_encode.conv.${idx}.bias`);
      allConvBiases.push(b!);
    }
  }

  // Output projection (may be `out.weight` or `out.0.weight`)
  let outWeight = tryGet('encoder.pre_encode.out.weight');
  let outBias = tryGet('encoder.pre_encode.out.bias');
  if (!outWeight) {
    outWeight = get('encoder.pre_encode.out.0.weight');
    outBias = tryGet('encoder.pre_encode.out.0.bias');
  }

  const subsampling: SubsamplingWeights = {
    allConvWeights,
    allConvBiases,
    outWeight: outWeight!,
    outBias: outBias!,
  };

  // === Encoder layers ===
  const layers: ConformerBlockWeights[] = [];
  for (let i = 0; i < config.encoderLayers; i++) {
    const prefix = `encoder.layers.${i}`;

    const ffn1: FeedForwardWeights = {
      norm: getLayerNorm(`${prefix}.norm_feed_forward1`),
      linear1: getLinear(`${prefix}.feed_forward1.linear1`),
      linear2: getLinear(`${prefix}.feed_forward1.linear2`),
    };

    const attn: AttentionWeights = {
      norm: getLayerNorm(`${prefix}.norm_self_att`),
      queryProj: getLinear(`${prefix}.self_attn.linear_q`),
      keyProj: getLinear(`${prefix}.self_attn.linear_k`),
      valueProj: getLinear(`${prefix}.self_attn.linear_v`),
      outProj: getLinear(`${prefix}.self_attn.linear_out`),
      posBiasU: get(`${prefix}.self_attn.pos_bias_u`),
      posBiasV: get(`${prefix}.self_attn.pos_bias_v`),
      posProj: tryGetLinear(`${prefix}.self_attn.linear_pos`),
    };

    // num_batches_tracked is a scalar tracking counter, not a model weight
    tryGet(`${prefix}.conv.batch_norm.num_batches_tracked`);

    const bnWeight = get(`${prefix}.conv.batch_norm.weight`);
    const bnBias = get(`${prefix}.conv.batch_norm.bias`);
    // Running stats may be absent (some checkpoints export only learned params).
    // Default to identity: mean=0, var=1 → batchnorm becomes scale+shift.
    let bnRunningMean = tryGet(`${prefix}.conv.batch_norm.running_mean`);
    let bnRunningVar = tryGet(`${prefix}.conv.batch_norm.running_var`);
    if ((!bnRunningMean || !bnRunningVar) && backend) {
      const bnDim = backend.getShape(bnWeight)[0] as number;
      bnRunningMean = bnRunningMean ?? backend.zeros([bnDim]);
      bnRunningVar = bnRunningVar ?? backend.ones([bnDim]);
    }
    if (!bnRunningMean || !bnRunningVar) {
      throw new Error(
        `Missing batch_norm running stats for ${prefix} and no backend provided to create defaults. ` +
        `Pass backend to mapWeights().`
      );
    }

    const conv: ConvModuleWeights = {
      norm: getLayerNorm(`${prefix}.norm_conv`),
      pointwise1Weight: get(`${prefix}.conv.pointwise_conv1.weight`),
      pointwise1Bias: tryGet(`${prefix}.conv.pointwise_conv1.bias`),
      depthwiseWeight: get(`${prefix}.conv.depthwise_conv.weight`),
      depthwiseBias: tryGet(`${prefix}.conv.depthwise_conv.bias`),
      batchNorm: {
        weight: bnWeight,
        bias: bnBias,
        runningMean: bnRunningMean,
        runningVar: bnRunningVar,
      },
      pointwise2Weight: get(`${prefix}.conv.pointwise_conv2.weight`),
      pointwise2Bias: tryGet(`${prefix}.conv.pointwise_conv2.bias`),
    };

    const ffn2: FeedForwardWeights = {
      norm: getLayerNorm(`${prefix}.norm_feed_forward2`),
      linear1: getLinear(`${prefix}.feed_forward2.linear1`),
      linear2: getLinear(`${prefix}.feed_forward2.linear2`),
    };

    const finalNorm = getLayerNorm(`${prefix}.norm_out`);
    layers.push({ ffn1, attn, conv, ffn2, finalNorm });
  }

  const encoderFinalNorm: LayerNormWeights | null =
    tryGet('encoder.norm.weight') ? getLayerNorm('encoder.norm') : null;

  // === Prediction network ===
  const predictionLayers = config.predNumLayers;
  const lstmWeightsIH: TensorHandle[] = [];
  const lstmWeightsHH: TensorHandle[] = [];
  const lstmBiasIH: TensorHandle[] = [];
  const lstmBiasHH: TensorHandle[] = [];
  for (let l = 0; l < predictionLayers; l++) {
    // Try both naming conventions
    let wih = tryGet(`decoder.prediction.dec_rnn.lstm.weight_ih_l${l}`);
    if (!wih) wih = get(`decoder.prediction.lstm.weight_ih_l${l}`);
    lstmWeightsIH.push(wih);

    let whh = tryGet(`decoder.prediction.dec_rnn.lstm.weight_hh_l${l}`);
    if (!whh) whh = get(`decoder.prediction.lstm.weight_hh_l${l}`);
    lstmWeightsHH.push(whh);

    let bih = tryGet(`decoder.prediction.dec_rnn.lstm.bias_ih_l${l}`);
    if (!bih) bih = get(`decoder.prediction.lstm.bias_ih_l${l}`);
    lstmBiasIH.push(bih);

    let bhh = tryGet(`decoder.prediction.dec_rnn.lstm.bias_hh_l${l}`);
    if (!bhh) bhh = get(`decoder.prediction.lstm.bias_hh_l${l}`);
    lstmBiasHH.push(bhh);
  }

  // Output projection may not exist in all models
  let predOutputProj = tryGetLinear('decoder.prediction.dec_rnn_layers.0');
  if (!predOutputProj) predOutputProj = tryGetLinear('decoder.prediction.dec_rnn.output_proj');

  const prediction: PredictionNetworkWeights = {
    embedding: get('decoder.prediction.embed.weight'),
    lstmWeightsIH,
    lstmWeightsHH,
    lstmBiasIH,
    lstmBiasHH,
    outputProj: predOutputProj,
  };

  // === Joint network ===
  // Try NeMo naming conventions
  let encProj = tryGetLinear('joint.enc');
  if (!encProj) encProj = tryGetLinear('joint.joint_net.0');
  if (!encProj) throw new Error('Missing joint encoder projection weights');

  let predProj = tryGetLinear('joint.pred');
  if (!predProj) predProj = tryGetLinear('joint.pred_net.0');
  if (!predProj) throw new Error('Missing joint prediction projection weights');

  const outputProj = getLinear('joint.joint_net.2');

  const joint: JointNetworkWeights = {
    encoderProj: encProj,
    predictionProj: predProj,
    outputProj,
  };

  // Mark known-skippable keys as consumed
  tryGet('preprocessor.featurizer.fb');
  tryGet('preprocessor.featurizer.window');
  // CTC decoder (hybrid models)
  for (const key of weights.keys()) {
    if (key.startsWith('ctc_decoder.')) {
      consumed.add(key);
    }
  }

  const unconsumed = [...weights.keys()].filter(k => !consumed.has(k) && k !== '__metadata__');
  if (unconsumed.length > 0) {
    console.warn('Unconsumed weights:', unconsumed);
  }

  return {
    encoder: { subsampling, layers, finalNorm: encoderFinalNorm },
    decoder: { prediction, joint },
  };
}
