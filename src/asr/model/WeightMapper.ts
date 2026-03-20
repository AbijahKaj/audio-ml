import type { TensorHandle } from '../compute/types';
import type { FastConformerConfig } from './ModelConfig';
import type {
  AttentionWeights,
  ConformerLayerWeights,
  DecoderWeights,
  EncoderWeights,
  FeedForwardWeights,
  JointWeights,
  LayerNormWeights,
  LinearWeights,
  ModelWeights,
  PredictionWeights,
} from './ModelWeights';

export interface WeightStudySummary {
  totalTensors: number;
  topLevelPrefixes: Array<{ prefix: string; count: number }>;
  largestTensors: Array<{ name: string; elements: number; shape: number[] }>;
}

function tensorShape(tensor: TensorHandle): number[] {
  const shape = (tensor as { shape?: number[] }).shape;
  return Array.isArray(shape) ? shape : [];
}

function elementCount(shape: number[]): number {
  if (shape.length === 0) return 0;
  return shape.reduce((acc, value) => acc * value, 1);
}

function prefixes(name: string): string {
  const parts = name.split('.');
  if (parts.length < 2) return parts[0];
  return `${parts[0]}.${parts[1]}`;
}

export function summarizeModelWeights(weights: Map<string, TensorHandle>): WeightStudySummary {
  const prefixCount = new Map<string, number>();
  const largest = [...weights.entries()]
    .map(([name, tensor]) => {
      const shape = tensorShape(tensor);
      return {
        name,
        shape,
        elements: elementCount(shape),
      };
    })
    .sort((a, b) => b.elements - a.elements)
    .slice(0, 20);

  for (const key of weights.keys()) {
    const prefix = prefixes(key);
    prefixCount.set(prefix, (prefixCount.get(prefix) ?? 0) + 1);
  }

  const topLevelPrefixes = [...prefixCount.entries()]
    .map(([prefix, count]) => ({ prefix, count }))
    .sort((a, b) => b.count - a.count);

  return {
    totalTensors: weights.size,
    topLevelPrefixes,
    largestTensors: largest,
  };
}

class WeightLookup {
  private consumed = new Set<string>();

  constructor(private source: Map<string, TensorHandle>) {}

  get consumedKeys(): string[] {
    return [...this.consumed.values()];
  }

  get unusedKeys(): string[] {
    return [...this.source.keys()].filter((key) => !this.consumed.has(key));
  }

  pick(aliases: string[], label: string): TensorHandle {
    for (const key of aliases) {
      const tensor = this.source.get(key);
      if (tensor) {
        this.consumed.add(key);
        return tensor;
      }
    }
    throw new Error(`Missing weight for ${label}. Tried aliases: ${aliases.join(', ')}`);
  }

  maybe(aliases: string[]): TensorHandle | undefined {
    for (const key of aliases) {
      const tensor = this.source.get(key);
      if (tensor) {
        this.consumed.add(key);
        return tensor;
      }
    }
    return undefined;
  }
}

function mapNorm(lookup: WeightLookup, names: string[]): LayerNormWeights {
  return {
    weight: lookup.pick(names.map((name) => `${name}.weight`), `${names[0]}.weight`),
    bias: lookup.pick(names.map((name) => `${name}.bias`), `${names[0]}.bias`),
  };
}

function mapLinear(lookup: WeightLookup, names: string[]): LinearWeights {
  return {
    weight: lookup.pick(names.map((name) => `${name}.weight`), `${names[0]}.weight`),
    bias: lookup.maybe(names.map((name) => `${name}.bias`)),
  };
}

function mapFeedForward(lookup: WeightLookup, layerIndex: number, first: boolean): FeedForwardWeights {
  const suffix = first ? '1' : '2';
  return {
    norm: mapNorm(lookup, [
      `encoder.layers.${layerIndex}.norm_feed_forward${suffix}`,
      `encoder.layers.${layerIndex}.norm_ff${suffix}`,
    ]),
    linear1: mapLinear(lookup, [
      `encoder.layers.${layerIndex}.fc${first ? '1' : '3'}`,
      `encoder.layers.${layerIndex}.feed_forward${suffix}.linear1`,
    ]),
    linear2: mapLinear(lookup, [
      `encoder.layers.${layerIndex}.fc${first ? '2' : '4'}`,
      `encoder.layers.${layerIndex}.feed_forward${suffix}.linear2`,
    ]),
  };
}

function mapAttention(lookup: WeightLookup, layerIndex: number): AttentionWeights {
  return {
    norm: mapNorm(lookup, [
      `encoder.layers.${layerIndex}.norm_self_att`,
      `encoder.layers.${layerIndex}.norm_self_attention`,
    ]),
    q: mapLinear(lookup, [
      `encoder.layers.${layerIndex}.self_attn.linear_q`,
      `encoder.layers.${layerIndex}.self_attn.q_proj`,
      `encoder.layers.${layerIndex}.self_attention.q_proj`,
    ]),
    k: mapLinear(lookup, [
      `encoder.layers.${layerIndex}.self_attn.linear_k`,
      `encoder.layers.${layerIndex}.self_attn.k_proj`,
      `encoder.layers.${layerIndex}.self_attention.k_proj`,
    ]),
    v: mapLinear(lookup, [
      `encoder.layers.${layerIndex}.self_attn.linear_v`,
      `encoder.layers.${layerIndex}.self_attn.v_proj`,
      `encoder.layers.${layerIndex}.self_attention.v_proj`,
    ]),
    out: mapLinear(lookup, [
      `encoder.layers.${layerIndex}.self_attn.linear_out`,
      `encoder.layers.${layerIndex}.self_attn.out_proj`,
      `encoder.layers.${layerIndex}.self_attention.out_proj`,
    ]),
    posBiasU: lookup.maybe([
      `encoder.layers.${layerIndex}.self_attn.pos_bias_u`,
      `encoder.layers.${layerIndex}.self_attention.pos_bias_u`,
    ]),
    posBiasV: lookup.maybe([
      `encoder.layers.${layerIndex}.self_attn.pos_bias_v`,
      `encoder.layers.${layerIndex}.self_attention.pos_bias_v`,
    ]),
  };
}

function mapConvModule(lookup: WeightLookup, layerIndex: number) {
  return {
    norm: mapNorm(lookup, [
      `encoder.layers.${layerIndex}.norm_conv`,
      `encoder.layers.${layerIndex}.norm_convolution`,
    ]),
    pointwise1: mapLinear(lookup, [
      `encoder.layers.${layerIndex}.conv.pointwise_conv1`,
      `encoder.layers.${layerIndex}.conv_module.pointwise_conv1`,
    ]),
    depthwiseKernel: lookup.pick(
      [
        `encoder.layers.${layerIndex}.conv.depthwise_conv.weight`,
        `encoder.layers.${layerIndex}.conv_module.depthwise_conv.weight`,
      ],
      `encoder.layers.${layerIndex}.depthwise_conv.weight`,
    ),
    batchNorm: {
      mean: lookup.pick(
        [
          `encoder.layers.${layerIndex}.conv.batch_norm.running_mean`,
          `encoder.layers.${layerIndex}.conv_module.batch_norm.running_mean`,
        ],
        `encoder.layers.${layerIndex}.batch_norm.running_mean`,
      ),
      variance: lookup.pick(
        [
          `encoder.layers.${layerIndex}.conv.batch_norm.running_var`,
          `encoder.layers.${layerIndex}.conv_module.batch_norm.running_var`,
        ],
        `encoder.layers.${layerIndex}.batch_norm.running_var`,
      ),
      scale: lookup.pick(
        [
          `encoder.layers.${layerIndex}.conv.batch_norm.weight`,
          `encoder.layers.${layerIndex}.conv_module.batch_norm.weight`,
        ],
        `encoder.layers.${layerIndex}.batch_norm.weight`,
      ),
      offset: lookup.pick(
        [
          `encoder.layers.${layerIndex}.conv.batch_norm.bias`,
          `encoder.layers.${layerIndex}.conv_module.batch_norm.bias`,
        ],
        `encoder.layers.${layerIndex}.batch_norm.bias`,
      ),
    },
    pointwise2: mapLinear(lookup, [
      `encoder.layers.${layerIndex}.conv.pointwise_conv2`,
      `encoder.layers.${layerIndex}.conv_module.pointwise_conv2`,
    ]),
  };
}

function mapConformerLayer(lookup: WeightLookup, layerIndex: number): ConformerLayerWeights {
  return {
    ffn1: mapFeedForward(lookup, layerIndex, true),
    attn: mapAttention(lookup, layerIndex),
    conv: mapConvModule(lookup, layerIndex),
    ffn2: mapFeedForward(lookup, layerIndex, false),
    finalNorm: mapNorm(lookup, [
      `encoder.layers.${layerIndex}.norm_out`,
      `encoder.layers.${layerIndex}.final_norm`,
    ]),
  };
}

function mapEncoder(lookup: WeightLookup, config: FastConformerConfig): EncoderWeights {
  return {
    subsampling: {
      conv1: mapLinear(lookup, ['encoder.pre_encode.conv.0', 'encoder.pre_encode.conv1']),
      conv2: mapLinear(lookup, ['encoder.pre_encode.conv.2', 'encoder.pre_encode.conv2']),
      conv3: lookup.maybe(['encoder.pre_encode.conv.4.weight', 'encoder.pre_encode.conv3.weight'])
        ? mapLinear(lookup, ['encoder.pre_encode.conv.4', 'encoder.pre_encode.conv3'])
        : undefined,
      out: mapLinear(lookup, ['encoder.pre_encode.out.0', 'encoder.pre_encode.out']),
    },
    layers: Array.from({ length: config.encoderLayers }, (_, layerIndex) => mapConformerLayer(lookup, layerIndex)),
  };
}

function mapPrediction(lookup: WeightLookup): PredictionWeights {
  return {
    embedding: lookup.pick(
      ['decoder.prediction.embed.weight', 'decoder.prediction.embedding.weight'],
      'decoder.prediction.embedding.weight',
    ),
    lstmWeightIH: lookup.pick(
      ['decoder.prediction.dec_rnn.weight_ih_l0', 'decoder.prediction.lstm.weight_ih_l0'],
      'decoder.prediction.lstm.weight_ih_l0',
    ),
    lstmWeightHH: lookup.pick(
      ['decoder.prediction.dec_rnn.weight_hh_l0', 'decoder.prediction.lstm.weight_hh_l0'],
      'decoder.prediction.lstm.weight_hh_l0',
    ),
    lstmBiasIH: lookup.pick(
      ['decoder.prediction.dec_rnn.bias_ih_l0', 'decoder.prediction.lstm.bias_ih_l0'],
      'decoder.prediction.lstm.bias_ih_l0',
    ),
    lstmBiasHH: lookup.pick(
      ['decoder.prediction.dec_rnn.bias_hh_l0', 'decoder.prediction.lstm.bias_hh_l0'],
      'decoder.prediction.lstm.bias_hh_l0',
    ),
    outputProj: mapLinear(lookup, [
      'decoder.prediction.output_proj',
      'decoder.prediction.project',
      'decoder.prediction.lin_out',
    ]),
  };
}

function mapJoint(lookup: WeightLookup, config: FastConformerConfig): JointWeights {
  const joint: JointWeights = {
    encoderProj: mapLinear(lookup, ['joint.enc', 'joint.encoder_proj', 'joint.joint_enc']),
    predictionProj: mapLinear(lookup, ['joint.pred', 'joint.prediction_proj', 'joint.joint_pred']),
    hiddenProj: mapLinear(lookup, ['joint.joint_net.0', 'joint.hidden']),
    tokenProj: mapLinear(lookup, ['joint.joint_net.2', 'joint.token_proj', 'joint.vocab_proj']),
  };

  if (config.decoderType === 'tdt') {
    joint.durationProj = mapLinear(lookup, ['joint.duration_head', 'joint.duration_proj']);
  }

  return joint;
}

function mapDecoder(lookup: WeightLookup, config: FastConformerConfig): DecoderWeights {
  return {
    prediction: mapPrediction(lookup),
    joint: mapJoint(lookup, config),
  };
}

export function mapWeights(
  weights: Map<string, TensorHandle>,
  config: FastConformerConfig,
): ModelWeights {
  const lookup = new WeightLookup(weights);
  const encoder = mapEncoder(lookup, config);
  const decoder = mapDecoder(lookup, config);

  return {
    encoder,
    decoder,
    consumedKeys: lookup.consumedKeys.sort(),
    unusedKeys: lookup.unusedKeys.sort(),
    raw: weights,
  };
}
