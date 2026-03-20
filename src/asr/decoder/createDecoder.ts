import type { ComputeBackend } from '../compute/Backend';
import type { FastConformerConfig } from '../model/ModelConfig';
import type { ModelWeights } from '../model/weights';
import { PredictionNetwork } from './PredictionNetwork';
import { RNNTGreedyDecoder } from './RNNTGreedyDecoder';
import { RNNTJointNetwork } from './RNNTJointNetwork';
import { TDTGreedyDecoder } from './TDTGreedyDecoder';
import { TDTJointNetwork } from './TDTJointNetwork';

export function createDecoder(
  config: FastConformerConfig,
  backend: ComputeBackend,
  weights: ModelWeights['decoder'],
): RNNTGreedyDecoder | TDTGreedyDecoder {
  const predNet = new PredictionNetwork(backend, weights.prediction, config.predHidden);
  if (config.decoderType === 'tdt') {
    const jointNet = new TDTJointNetwork(backend, weights.joint);
    return new TDTGreedyDecoder(backend, predNet, jointNet, config.blankTokenId, config);
  }
  const jointNet = new RNNTJointNetwork(backend, weights.joint);
  return new RNNTGreedyDecoder(backend, predNet, jointNet, config.blankTokenId);
}
