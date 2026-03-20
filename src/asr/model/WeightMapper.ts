import type { ComputeBackend } from '../compute/Backend';
import type { TensorHandle } from '../compute/types';
import type { FastConformerConfig } from './ModelConfig';
import type {
  BatchNormWeights,
  ConformerConvWeights,
  ConformerLayerWeights,
  DecoderWeights,
  EncoderWeights,
  LinearWeights,
  ModelWeights,
  PredictionWeights,
  RelPosAttnWeights,
  RNNTJointWeights,
  SubsamplingWeights,
  TDTJointWeights,
} from './types';

function getTensor(weights: Map<string, TensorHandle>, keys: string[]): TensorHandle {
  for (const k of keys) {
    if (weights.has(k)) {
      return weights.get(k)!;
    }
  }
  throw new Error(`Missing weight; tried: ${keys.join(', ')}`);
}

function linearAt(weights: Map<string, TensorHandle>, modulePath: string): LinearWeights {
  return {
    weight: getTensor(weights, [`${modulePath}.weight`]),
    bias: getTensor(weights, [`${modulePath}.bias`]),
  };
}

function layerNormAt(weights: Map<string, TensorHandle>, modulePath: string): {
  weight: TensorHandle;
  bias: TensorHandle;
} {
  return {
    weight: getTensor(weights, [`${modulePath}.weight`]),
    bias: getTensor(weights, [`${modulePath}.bias`]),
  };
}

function buildSubsampling(weights: Map<string, TensorHandle>, encRoot: string): SubsamplingWeights {
  const pre = `${encRoot}encoder.pre_encode`;
  return {
    conv0: {
      weight: getTensor(weights, [`${pre}.conv.0.weight`]),
      bias: getTensor(weights, [`${pre}.conv.0.bias`]),
    },
    conv1: {
      weight: getTensor(weights, [`${pre}.conv.2.weight`]),
      bias: getTensor(weights, [`${pre}.conv.2.bias`]),
    },
    conv2: {
      weight: getTensor(weights, [`${pre}.conv.4.weight`]),
      bias: getTensor(weights, [`${pre}.conv.4.bias`]),
    },
    out: linearAt(weights, `${pre}.out`),
  };
}

function buildConvModule(weights: Map<string, TensorHandle>, prefix: string): ConformerConvWeights {
  const bn = `${prefix}.batch_norm`;
  const batchNorm: BatchNormWeights = {
    runningMean: getTensor(weights, [`${bn}.running_mean`]),
    runningVar: getTensor(weights, [`${bn}.running_var`]),
    weight: getTensor(weights, [`${bn}.weight`]),
    bias: getTensor(weights, [`${bn}.bias`]),
  };
  return {
    pointwise1: linearAt(weights, `${prefix}.pointwise_conv1`),
    depthwise: getTensor(weights, [`${prefix}.depthwise_conv.weight`]),
    depthwiseBias: getTensor(weights, [`${prefix}.depthwise_conv.bias`]),
    batchNorm,
    pointwise2: linearAt(weights, `${prefix}.pointwise_conv2`),
  };
}

function buildSelfAttn(weights: Map<string, TensorHandle>, prefix: string): RelPosAttnWeights {
  return {
    linearQ: linearAt(weights, `${prefix}.linear_q`),
    linearK: linearAt(weights, `${prefix}.linear_k`),
    linearV: linearAt(weights, `${prefix}.linear_v`),
    linearOut: linearAt(weights, `${prefix}.linear_out`),
    linearPos: linearAt(weights, `${prefix}.linear_pos`),
    posBiasU: getTensor(weights, [`${prefix}.pos_bias_u`]),
    posBiasV: getTensor(weights, [`${prefix}.pos_bias_v`]),
  };
}

function buildLayer(weights: Map<string, TensorHandle>, i: number, encRoot: string): ConformerLayerWeights {
  const p = `${encRoot}encoder.layers.${i}`;
  return {
    normFeedForward1: layerNormAt(weights, `${p}.norm_feed_forward1`),
    feedForward1: {
      w1: linearAt(weights, `${p}.feed_forward1.linear1`),
      w2: linearAt(weights, `${p}.feed_forward1.linear2`),
    },
    normSelfAtt: layerNormAt(weights, `${p}.norm_self_att`),
    selfAtt: buildSelfAttn(weights, `${p}.self_attn`),
    normConv: layerNormAt(weights, `${p}.norm_conv`),
    conv: buildConvModule(weights, `${p}.conv`),
    normFeedForward2: layerNormAt(weights, `${p}.norm_feed_forward2`),
    feedForward2: {
      w1: linearAt(weights, `${p}.feed_forward2.linear1`),
      w2: linearAt(weights, `${p}.feed_forward2.linear2`),
    },
    normOut: layerNormAt(weights, `${p}.norm_out`),
  };
}

function buildEncoderWeights(weights: Map<string, TensorHandle>, config: FastConformerConfig): EncoderWeights {
  const encRoot = config.stateDictPrefix ?? '';
  const subsampling = buildSubsampling(weights, encRoot);
  const layers: ConformerLayerWeights[] = [];
  for (let i = 0; i < config.encoderLayers; i++) {
    layers.push(buildLayer(weights, i, encRoot));
  }
  return { subsampling, layers };
}

function buildPrediction(weights: Map<string, TensorHandle>, decRoot: string, backend: ComputeBackend): PredictionWeights {
  const pred = `${decRoot}decoder.prediction.predictions`;
  const bih = getTensor(weights, [`${pred}.decoder.bias_ih_l0`]);
  const bhh = getTensor(weights, [`${pred}.decoder.bias_hh_l0`]);
  const lstmBias = backend.add(bih, bhh);
  backend.dispose(bih);
  backend.dispose(bhh);
  return {
    embedding: getTensor(weights, [`${pred}.embedding.weight`]),
    lstmWih: getTensor(weights, [`${pred}.decoder.weight_ih_l0`]),
    lstmWhh: getTensor(weights, [`${pred}.decoder.weight_hh_l0`]),
    lstmBias,
  };
}

function buildJointOutput(weights: Map<string, TensorHandle>, jointBase: string): LinearWeights {
  const tries = [`${jointBase}.joint_net.2`, `${jointBase}.joint_net.1`, `${jointBase}.joint_net.3`];
  for (const path of tries) {
    if (weights.has(`${path}.weight`)) {
      return linearAt(weights, path);
    }
  }
  throw new Error(`Could not find joint output Linear near ${jointBase}.joint_net.*`);
}

function buildRnntJoint(weights: Map<string, TensorHandle>, decRoot: string): RNNTJointWeights {
  const jb = `${decRoot}decoder.joint`;
  return {
    encoderProj: linearAt(weights, `${jb}.enc`),
    predProj: linearAt(weights, `${jb}.pred`),
    output: buildJointOutput(weights, jb),
  };
}

function buildTdtJoint(weights: Map<string, TensorHandle>, decRoot: string): TDTJointWeights {
  const base = buildRnntJoint(weights, decRoot) as TDTJointWeights;
  const jb = `${decRoot}decoder.joint`;
  const durTries = [`${jb}.duration`, `${jb}.duration_head`, `${jb}.duration_proj`];
  for (const p of durTries) {
    if (weights.has(`${p}.weight`)) {
      base.duration = linearAt(weights, p);
      return base;
    }
  }
  throw new Error('TDT duration head not found on joint module');
}

function buildDecoderWeights(
  weights: Map<string, TensorHandle>,
  config: FastConformerConfig,
  backend: ComputeBackend,
): DecoderWeights {
  const decRoot = config.decoderStateDictPrefix ?? '';
  const prediction = buildPrediction(weights, decRoot, backend);
  if (config.decoderType === 'tdt') {
    return { prediction, joint: buildTdtJoint(weights, decRoot) };
  }
  return { prediction, joint: buildRnntJoint(weights, decRoot) };
}

/**
 * Map a flat NeMo-style state dict (SafeTensors keys) into structured {@link ModelWeights}.
 */
export function mapWeights(
  weights: Map<string, TensorHandle>,
  config: FastConformerConfig,
  backend: ComputeBackend,
): ModelWeights {
  if (config.selfAttentionModel === 'rel_pos_local_attn') {
    throw new Error(
      'rel_pos_local_attn checkpoints require Longformer-style attention (not implemented). Use rel_pos exports or a compatible checkpoint.',
    );
  }
  return {
    encoder: buildEncoderWeights(weights, config),
    decoder: buildDecoderWeights(weights, config, backend),
  };
}