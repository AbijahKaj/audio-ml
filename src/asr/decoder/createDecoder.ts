import type { ComputeBackend } from '../compute/Backend.js';
import type { FastConformerConfig } from '../model/ModelConfig.js';
import type { DecoderWeights } from '../model/WeightMapper.js';
import { PredictionNetwork } from './PredictionNetwork.js';
import { RNNTGreedyDecoder } from './RNNTGreedyDecoder.js';
import { RNNTJointNetwork } from './RNNTJointNetwork.js';
import { TDTGreedyDecoder } from './TDTGreedyDecoder.js';
import { TDTJointNetwork } from './TDTJointNetwork.js';

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
