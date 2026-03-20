import type { ComputeBackend } from '../compute/Backend';
import type { FastConformerConfig } from '../model/ModelConfig';
import type { DecoderWeights } from '../model/WeightMapper';
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
  const prediction = new PredictionNetwork(backend, weights.prediction, config.predHidden);

  if (config.decoderType === 'tdt') {
    return new TDTGreedyDecoder(backend, config, prediction, new TDTJointNetwork(backend, weights.joint));
  }

  return new RNNTGreedyDecoder(backend, config, prediction, new RNNTJointNetwork(backend, weights.joint));
}
