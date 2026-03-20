import type { ComputeBackend } from '../compute/Backend';
import type { DecoderWeights } from '../model/WeightMapper';
import type { FastConformerConfig } from '../model/ModelConfig';
import { PredictionNetwork } from './PredictionNetwork';
import { RNNTJointNetwork } from './RNNTJointNetwork';
import { TDTJointNetwork } from './TDTJointNetwork';
import { RNNTGreedyDecoder } from './RNNTGreedyDecoder';
import { TDTGreedyDecoder } from './TDTGreedyDecoder';

export type GreedyDecoder = RNNTGreedyDecoder | TDTGreedyDecoder;

export function createDecoder(
  config: FastConformerConfig,
  backend: ComputeBackend,
  weights: DecoderWeights,
): GreedyDecoder {
  const predNet = new PredictionNetwork(backend, weights.prediction, config.predHidden);

  if (config.decoderType === 'tdt') {
    const jointNet = new TDTJointNetwork(backend, weights.joint);
    return new TDTGreedyDecoder(
      backend,
      predNet,
      jointNet,
      config.tdtDurations ?? [0, 1, 2, 3, 4],
    );
  }

  const jointNet = new RNNTJointNetwork(backend, weights.joint);
  return new RNNTGreedyDecoder(backend, predNet, jointNet);
}
