import type { ComputeBackend } from '../compute/index.js';
import type { DecoderWeights } from '../model/WeightMapper.js';
import type { FastConformerConfig } from '../model/ModelConfig.js';
import { PredictionNetwork } from './PredictionNetwork.js';
import { RNNTJointNetwork } from './RNNTJointNetwork.js';
import { TDTJointNetwork } from './TDTJointNetwork.js';
import { RNNTGreedyDecoder } from './RNNTGreedyDecoder.js';
import { TDTGreedyDecoder } from './TDTGreedyDecoder.js';

/**
 * Factory: creates the correct greedy decoder based on model config.
 *
 * - config.decoderType === 'rnnt' → RNNTGreedyDecoder
 * - config.decoderType === 'tdt'  → TDTGreedyDecoder
 *
 * Both use the same PredictionNetwork — only the joint network and decode
 * loop differ.
 */
export function createDecoder(
  config: FastConformerConfig,
  backend: ComputeBackend,
  weights: DecoderWeights,
): RNNTGreedyDecoder | TDTGreedyDecoder {
  const predNet = new PredictionNetwork(backend, weights.prediction);

  if (config.decoderType === 'tdt') {
    const jointNet = new TDTJointNetwork(backend, weights.joint);
    const durations = config.tdtDurations ?? [0, 1, 2, 3, 4];
    return new TDTGreedyDecoder(backend, predNet, jointNet, 0, durations);
  }

  const jointNet = new RNNTJointNetwork(backend, weights.joint);
  return new RNNTGreedyDecoder(backend, predNet, jointNet);
}
