import type { ComputeBackend } from '../compute/Backend';
import type { FastConformerConfig } from '../model/ModelConfig';
import type { DecoderWeights } from '../model/ModelWeights';
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
  const prediction = new PredictionNetwork(backend, weights.prediction);

  if (config.decoderType === 'tdt') {
    const durations = config.tdtDurations ?? Array.from({ length: config.tdtNumDurations ?? 5 }, (_, i) => i);
    const joint = new TDTJointNetwork(backend, weights.joint);
    return new TDTGreedyDecoder(backend, prediction, joint, durations, config.blankId);
  }

  const joint = new RNNTJointNetwork(backend, weights.joint);
  return new RNNTGreedyDecoder(backend, prediction, joint, config.blankId);
}
