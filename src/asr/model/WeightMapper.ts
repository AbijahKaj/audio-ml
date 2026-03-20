import type { TensorHandle } from '../compute/types';
import type { FastConformerConfig } from './ModelConfig';

export interface LinearWeightPair {
  weight: TensorHandle | null;
  bias: TensorHandle | null;
}

export interface LayerNormWeightPair {
  weight: TensorHandle | null;
  bias: TensorHandle | null;
}

export interface FeedForwardWeights {
  norm: LayerNormWeightPair;
  linear1: LinearWeightPair;
  linear2: LinearWeightPair;
}

export interface AttentionWeights {
  norm: LayerNormWeightPair;
  qProj: LinearWeightPair;
  kProj: LinearWeightPair;
  vProj: LinearWeightPair;
  outProj: LinearWeightPair;
}

export interface ConvModuleWeights {
  norm: LayerNormWeightPair;
  pointwiseIn: LinearWeightPair;
  depthwiseWeight: TensorHandle | null;
  batchNormMean: TensorHandle | null;
  batchNormVariance: TensorHandle | null;
  batchNormScale: TensorHandle | null;
  batchNormOffset: TensorHandle | null;
  pointwiseOut: LinearWeightPair;
}

export interface ConformerLayerWeights {
  ffn1: FeedForwardWeights;
  attention: AttentionWeights;
  conv: ConvModuleWeights;
  ffn2: FeedForwardWeights;
  finalNorm: LayerNormWeightPair;
}

export interface EncoderWeights {
  subsamplingIn: LinearWeightPair;
  subsamplingOut: LinearWeightPair;
  layers: ConformerLayerWeights[];
}

export interface PredictionWeights {
  embedding: TensorHandle | null;
  lstmWeightIH: TensorHandle | null;
  lstmWeightHH: TensorHandle | null;
  lstmBiasIH: TensorHandle | null;
  lstmBiasHH: TensorHandle | null;
  outputProj: LinearWeightPair;
}

export interface JointWeights {
  encoderProj: LinearWeightPair;
  predictionProj: LinearWeightPair;
  outputProj: LinearWeightPair;
  durationProj: LinearWeightPair;
}

export interface DecoderWeights {
  prediction: PredictionWeights;
  joint: JointWeights;
}

export interface ModelWeights {
  encoder: EncoderWeights;
  decoder: DecoderWeights;
  raw: Map<string, TensorHandle>;
  unconsumed: string[];
}

function pickFirst(
  weights: Map<string, TensorHandle>,
  consumed: Set<string>,
  names: string[],
): TensorHandle | null {
  for (const name of names) {
    const tensor = weights.get(name);
    if (tensor) {
      consumed.add(name);
      return tensor;
    }
  }
  return null;
}

function linear(
  weights: Map<string, TensorHandle>,
  consumed: Set<string>,
  weightNames: string[],
  biasNames: string[],
): LinearWeightPair {
  return {
    weight: pickFirst(weights, consumed, weightNames),
    bias: pickFirst(weights, consumed, biasNames),
  };
}

function norm(
  weights: Map<string, TensorHandle>,
  consumed: Set<string>,
  prefix: string,
): LayerNormWeightPair {
  return {
    weight: pickFirst(weights, consumed, [`${prefix}.weight`, `${prefix}.gamma`]),
    bias: pickFirst(weights, consumed, [`${prefix}.bias`, `${prefix}.beta`]),
  };
}

export function mapWeights(
  weights: Map<string, TensorHandle>,
  config: FastConformerConfig,
): ModelWeights {
  const consumed = new Set<string>();
  const layers: ConformerLayerWeights[] = [];

  for (let index = 0; index < config.encoderLayers; index += 1) {
    const prefix = `encoder.layers.${index}`;
    layers.push({
      ffn1: {
        norm: norm(weights, consumed, `${prefix}.norm_feed_forward1`),
        linear1: linear(weights, consumed, [`${prefix}.feed_forward1.linear1.weight`, `${prefix}.fc1.weight`], [`${prefix}.feed_forward1.linear1.bias`, `${prefix}.fc1.bias`]),
        linear2: linear(weights, consumed, [`${prefix}.feed_forward1.linear2.weight`, `${prefix}.fc2.weight`], [`${prefix}.feed_forward1.linear2.bias`, `${prefix}.fc2.bias`]),
      },
      attention: {
        norm: norm(weights, consumed, `${prefix}.norm_self_att`),
        qProj: linear(weights, consumed, [`${prefix}.self_attn.linear_q.weight`, `${prefix}.self_attn.q_proj.weight`], [`${prefix}.self_attn.linear_q.bias`, `${prefix}.self_attn.q_proj.bias`]),
        kProj: linear(weights, consumed, [`${prefix}.self_attn.linear_k.weight`, `${prefix}.self_attn.k_proj.weight`], [`${prefix}.self_attn.linear_k.bias`, `${prefix}.self_attn.k_proj.bias`]),
        vProj: linear(weights, consumed, [`${prefix}.self_attn.linear_v.weight`, `${prefix}.self_attn.v_proj.weight`], [`${prefix}.self_attn.linear_v.bias`, `${prefix}.self_attn.v_proj.bias`]),
        outProj: linear(weights, consumed, [`${prefix}.self_attn.linear_out.weight`, `${prefix}.self_attn.out_proj.weight`], [`${prefix}.self_attn.linear_out.bias`, `${prefix}.self_attn.out_proj.bias`]),
      },
      conv: {
        norm: norm(weights, consumed, `${prefix}.norm_conv`),
        pointwiseIn: linear(weights, consumed, [`${prefix}.conv.pointwise_conv1.weight`, `${prefix}.conv.pointwise1.weight`], [`${prefix}.conv.pointwise_conv1.bias`, `${prefix}.conv.pointwise1.bias`]),
        depthwiseWeight: pickFirst(weights, consumed, [`${prefix}.conv.depthwise_conv.weight`, `${prefix}.conv.depthwise.weight`]),
        batchNormMean: pickFirst(weights, consumed, [`${prefix}.conv.batch_norm.running_mean`]),
        batchNormVariance: pickFirst(weights, consumed, [`${prefix}.conv.batch_norm.running_var`]),
        batchNormScale: pickFirst(weights, consumed, [`${prefix}.conv.batch_norm.weight`]),
        batchNormOffset: pickFirst(weights, consumed, [`${prefix}.conv.batch_norm.bias`]),
        pointwiseOut: linear(weights, consumed, [`${prefix}.conv.pointwise_conv2.weight`, `${prefix}.conv.pointwise2.weight`], [`${prefix}.conv.pointwise_conv2.bias`, `${prefix}.conv.pointwise2.bias`]),
      },
      ffn2: {
        norm: norm(weights, consumed, `${prefix}.norm_feed_forward2`),
        linear1: linear(weights, consumed, [`${prefix}.feed_forward2.linear1.weight`, `${prefix}.fc3.weight`], [`${prefix}.feed_forward2.linear1.bias`, `${prefix}.fc3.bias`]),
        linear2: linear(weights, consumed, [`${prefix}.feed_forward2.linear2.weight`, `${prefix}.fc4.weight`], [`${prefix}.feed_forward2.linear2.bias`, `${prefix}.fc4.bias`]),
      },
      finalNorm: norm(weights, consumed, `${prefix}.norm_out`),
    });
  }

  const encoder: EncoderWeights = {
    subsamplingIn: linear(
      weights,
      consumed,
      ['encoder.pre_encode.out.0.weight', 'encoder.pre_encode.linear.weight'],
      ['encoder.pre_encode.out.0.bias', 'encoder.pre_encode.linear.bias'],
    ),
    subsamplingOut: linear(
      weights,
      consumed,
      ['encoder.pre_encode.out.2.weight', 'encoder.out_proj.weight'],
      ['encoder.pre_encode.out.2.bias', 'encoder.out_proj.bias'],
    ),
    layers,
  };

  const decoder: DecoderWeights = {
    prediction: {
      embedding: pickFirst(weights, consumed, ['decoder.prediction.embed.weight', 'decoder.prediction.embedding.weight']),
      lstmWeightIH: pickFirst(weights, consumed, ['decoder.prediction.dec_rnn.weight_ih_l0']),
      lstmWeightHH: pickFirst(weights, consumed, ['decoder.prediction.dec_rnn.weight_hh_l0']),
      lstmBiasIH: pickFirst(weights, consumed, ['decoder.prediction.dec_rnn.bias_ih_l0']),
      lstmBiasHH: pickFirst(weights, consumed, ['decoder.prediction.dec_rnn.bias_hh_l0']),
      outputProj: linear(weights, consumed, ['decoder.prediction.project.weight'], ['decoder.prediction.project.bias']),
    },
    joint: {
      encoderProj: linear(weights, consumed, ['joint.enc.weight', 'joint.encoder_proj.weight'], ['joint.enc.bias', 'joint.encoder_proj.bias']),
      predictionProj: linear(weights, consumed, ['joint.pred.weight', 'joint.prediction_proj.weight'], ['joint.pred.bias', 'joint.prediction_proj.bias']),
      outputProj: linear(weights, consumed, ['joint.joint_net.2.weight', 'joint.output.weight'], ['joint.joint_net.2.bias', 'joint.output.bias']),
      durationProj: linear(weights, consumed, ['joint.duration_head.weight', 'joint.duration_proj.weight'], ['joint.duration_head.bias', 'joint.duration_proj.bias']),
    },
  };

  return {
    encoder,
    decoder,
    raw: weights,
    unconsumed: [...weights.keys()].filter((name) => !consumed.has(name)),
  };
}
