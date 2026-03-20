import type { ComputeBackend } from '../compute/Backend';
import type { FastConformerConfig } from './ModelConfig';
import type {
  ConformerLayerWeights,
  ConvModuleWeights,
  EncoderWeights,
  JointWeights,
  LayerNormParams,
  LinearParams,
  ModelWeights,
  PredictionWeights,
  SelfAttentionWeights,
  SubsamplingLayerSpec,
  SubsamplingWeights,
} from './weights';
import type { TensorHandle } from '../compute/types';

function ln(get: (k: string) => TensorHandle, w: string, b: string): LayerNormParams {
  return { weight: get(w), bias: get(b) };
}

function linearParam(
  weights: Map<string, TensorHandle>,
  get: (k: string) => TensorHandle,
  w: string,
  expectBias = true,
): LinearParams {
  const weight = get(w);
  const biasKey = w.replace('.weight', '.bias');
  if (!expectBias || !weights.has(biasKey)) {
    return { weight, bias: null };
  }
  return { weight, bias: get(biasKey) };
}

function collectSubsamplingConvKeys(weights: Map<string, TensorHandle>): number[] {
  const idx = new Set<number>();
  for (const k of weights.keys()) {
    const m = k.match(/^encoder\.pre_encode\.conv\.(\d+)\.weight$/);
    if (m) {
      idx.add(parseInt(m[1], 10));
    }
  }
  return [...idx].sort((a, b) => a - b);
}

function jointNetModulePrefix(weights: Map<string, TensorHandle>): string {
  const keys = [...weights.keys()].filter(k => /^joint\.joint_net\.\d+\.weight$/.test(k));
  if (keys.length === 0) {
    throw new Error('No joint.joint_net.*.weight in checkpoint');
  }
  const last = keys
    .map(k => parseInt(k.split('.')[2], 10))
    .sort((a, b) => b - a)[0];
  return `joint.joint_net.${last}`;
}

function mapSubsampling(
  get: (k: string) => TensorHandle,
  weights: Map<string, TensorHandle>,
  backend: ComputeBackend,
  config: FastConformerConfig,
): SubsamplingWeights {
  const convIndices = collectSubsamplingConvKeys(weights);
  const layers: SubsamplingLayerSpec[] = [];
  const subs = config.subsampling;

  for (const i of convIndices) {
    const weight = get(`encoder.pre_encode.conv.${i}.weight`);
    const bias = weights.has(`encoder.pre_encode.conv.${i}.bias`)
      ? get(`encoder.pre_encode.conv.${i}.bias`)
      : null;
    const sh = backend.getShape(weight);

    if (subs !== 'dw_striding' || sh.length !== 4) {
      layers.push({ kind: 'conv2d_s2', weight, bias });
      continue;
    }

    const [, inCh, kh, kw] = sh;
    if (layers.length === 0 && inCh === 1 && kh === 3) {
      layers.push({ kind: 'conv2d_s2', weight, bias });
    } else if (inCh === 1 && kh === 3) {
      layers.push({ kind: 'depthwise_s2', weight, bias });
    } else if (kh === 1 && kw === 1) {
      layers.push({ kind: 'pointwise1x1', weight, bias });
    } else {
      layers.push({ kind: 'conv2d_s2', weight, bias });
    }
  }

  const outW = linearParam(weights, get, 'encoder.pre_encode.out.weight');
  return { layers, out: outW };
}

function mapConvModule(
  weights: Map<string, TensorHandle>,
  get: (k: string) => TensorHandle,
  prefix: string,
): ConvModuleWeights {
  const useBn = weights.has(`${prefix}.batch_norm.running_mean`);
  return {
    afterDepthwise: ln(get, `${prefix}.batch_norm.weight`, `${prefix}.batch_norm.bias`),
    useBatchNormStats: useBn,
    bnRunningMean: useBn ? get(`${prefix}.batch_norm.running_mean`) : null,
    bnRunningVar: useBn ? get(`${prefix}.batch_norm.running_var`) : null,
    pointwise1: linearParam(weights, get, `${prefix}.pointwise_conv1.weight`, false),
    depthwiseWeight: get(`${prefix}.depthwise_conv.weight`),
    depthwiseBias: weights.has(`${prefix}.depthwise_conv.bias`)
      ? get(`${prefix}.depthwise_conv.bias`)
      : null,
    pointwise2: linearParam(weights, get, `${prefix}.pointwise_conv2.weight`, false),
  };
}

function mapSelfAtt(
  weights: Map<string, TensorHandle>,
  get: (k: string) => TensorHandle,
  prefix: string,
): SelfAttentionWeights {
  return {
    linearQ: linearParam(weights, get, `${prefix}.linear_q.weight`),
    linearK: linearParam(weights, get, `${prefix}.linear_k.weight`),
    linearV: linearParam(weights, get, `${prefix}.linear_v.weight`),
    linearOut: linearParam(weights, get, `${prefix}.linear_out.weight`),
    linearPos: linearParam(weights, get, `${prefix}.linear_pos.weight`, false),
    posBiasU: get(`${prefix}.pos_bias_u`),
    posBiasV: get(`${prefix}.pos_bias_v`),
  };
}

function mapConformerLayer(
  weights: Map<string, TensorHandle>,
  get: (k: string) => TensorHandle,
  i: number,
): ConformerLayerWeights {
  const p = `encoder.layers.${i}`;
  return {
    normFeedForward1: ln(get, `${p}.norm_feed_forward1.weight`, `${p}.norm_feed_forward1.bias`),
    feedForward1: {
      linear1: linearParam(weights, get, `${p}.feed_forward1.linear1.weight`),
      linear2: linearParam(weights, get, `${p}.feed_forward1.linear2.weight`),
    },
    normSelfAtt: ln(get, `${p}.norm_self_att.weight`, `${p}.norm_self_att.bias`),
    selfAtt: mapSelfAtt(weights, get, `${p}.self_attn`),
    normConv: ln(get, `${p}.norm_conv.weight`, `${p}.norm_conv.bias`),
    conv: mapConvModule(weights, get, `${p}.conv`),
    normFeedForward2: ln(get, `${p}.norm_feed_forward2.weight`, `${p}.norm_feed_forward2.bias`),
    feedForward2: {
      linear1: linearParam(weights, get, `${p}.feed_forward2.linear1.weight`),
      linear2: linearParam(weights, get, `${p}.feed_forward2.linear2.weight`),
    },
    normOut: ln(get, `${p}.norm_out.weight`, `${p}.norm_out.bias`),
  };
}

function mapPrediction(get: (k: string) => TensorHandle, weights: Map<string, TensorHandle>): PredictionWeights {
  const embedKey = [...weights.keys()].find(
    k => k === 'decoder.prediction.embed.weight' || k.endsWith('prediction.embed.weight'),
  );
  if (!embedKey) {
    throw new Error('Missing decoder.prediction.embed.weight');
  }
  const embedding = get(embedKey);

  const lstmLayerIdx = new Set<number>();
  for (const k of weights.keys()) {
    const m = k.match(/^decoder\.prediction\.dec_rnn(?:\.lstm)?\.weight_ih_l(\d+)$/);
    if (m) {
      lstmLayerIdx.add(parseInt(m[1], 10));
    }
  }
  const layers = [...lstmLayerIdx].sort((a, b) => a - b);
  if (layers.length === 0) {
    throw new Error('No decoder.prediction.dec_rnn LSTM weights');
  }
  const lstmPrefix = weights.has('decoder.prediction.dec_rnn.lstm.weight_ih_l0')
    ? 'decoder.prediction.dec_rnn.lstm'
    : 'decoder.prediction.dec_rnn';
  const lstm = {
    weightIh: get(`${lstmPrefix}.weight_ih_l0`),
    weightHh: get(`${lstmPrefix}.weight_hh_l0`),
    biasIh: get(`${lstmPrefix}.bias_ih_l0`),
    biasHh: get(`${lstmPrefix}.bias_hh_l0`),
  };
  if (layers.length > 1) {
    console.warn(
      `WeightMapper: dec_rnn has ${layers.length} layers; using layer 0 only (extend PredictionNetwork for full support).`,
    );
  }
  return { embedding, lstm, numLayers: layers.length };
}

function mapJoint(
  get: (k: string) => TensorHandle,
  weights: Map<string, TensorHandle>,
  config: FastConformerConfig,
): JointWeights {
  const prefix = jointNetModulePrefix(weights);
  const jointNetLinear = linearParam(weights, get, `${prefix}.weight`);
  const j: JointWeights = {
    enc: linearParam(weights, get, 'joint.enc.weight'),
    pred: linearParam(weights, get, 'joint.pred.weight'),
    jointNetLinear,
  };
  if (config.decoderType === 'tdt') {
    const dur = [...weights.keys()].find(k => k.includes('duration') && k.endsWith('.weight'));
    if (dur) {
      j.durationLinear = linearParam(weights, get, dur);
    }
  }
  return j;
}

export function mapWeights(
  weights: Map<string, TensorHandle>,
  config: FastConformerConfig,
  backend: ComputeBackend,
): ModelWeights {
  const consumed = new Set<string>();

  function get(name: string): TensorHandle {
    if (!weights.has(name)) {
      throw new Error(`Missing weight: ${name}`);
    }
    consumed.add(name);
    return weights.get(name)!;
  }

  const encoder: EncoderWeights = {
    subsampling: mapSubsampling(get, weights, backend, config),
    layers: Array.from({ length: config.encoderLayers }, (_, i) => mapConformerLayer(weights, get, i)),
  };

  const decoder = {
    prediction: mapPrediction(get, weights),
    joint: mapJoint(get, weights, config),
  };

  const unconsumed = [...weights.keys()].filter(k => !consumed.has(k));
  if (unconsumed.length > 0) {
    console.warn(`Unconsumed weights (${unconsumed.length}):`, unconsumed.slice(0, 20), unconsumed.length > 20 ? '...' : '');
  }

  return { encoder, decoder };
}
