import type { ComputeBackend } from '../compute/Backend';
import type { FastConformerConfig } from '../model/ModelConfig';
import type { DecoderWeights, RNNTJointWeights, TDTJointWeights } from '../model/types';
import { PredictionNetwork } from './PredictionNetwork';
import { RNNTGreedyDecoder } from './RNNTGreedyDecoder';
import { RNNTJointNetwork } from './RNNTJointNetwork';
import { TDTGreedyDecoder } from './TDTGreedyDecoder';
import { TDTJointNetwork } from './TDTJointNetwork';

export function createDecoder(
  config: FastConformerConfig,
  backend: ComputeBackend,
  weights: DecoderWeights,
): RNNTGreedyDecoder | TDTGreedyDecoder {
  const predNet = new PredictionNetwork(backend, weights.prediction);

  if (config.decoderType === 'tdt') {
    const jointNet = new TDTJointNetwork(backend, weights.joint as TDTJointWeights);
    return new TDTGreedyDecoder(backend, predNet, jointNet, config);
  }

  const jointNet = new RNNTJointNetwork(backend, weights.joint as RNNTJointWeights);
  return new RNNTGreedyDecoder(backend, predNet, jointNet, config);
}