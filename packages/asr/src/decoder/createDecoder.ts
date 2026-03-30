import type { ComputeBackend } from '../compute/Backend';
import type { DecoderWeights } from '../model/WeightMapper';
import type { FastConformerConfig } from '../model/ModelConfig';
import { PredictionNetwork } from './PredictionNetwork';
import { TDTJointNetwork } from './TDTJointNetwork';
import { TDTGreedyDecoder } from './TDTGreedyDecoder';

export type GreedyDecoder = TDTGreedyDecoder;

export function createDecoder(
  config: FastConformerConfig,
  backend: ComputeBackend,
  weights: DecoderWeights,
): GreedyDecoder {
  const predNet = new PredictionNetwork(backend, weights.prediction, config.predHidden);
  // NeMo convention: blank is the last token (appended after the BPE vocabulary)
  const blankId = config.vocabSize - 1;

  const durations = config.tdtDurations ?? [0, 1, 2, 3, 4];
  const jointNet = new TDTJointNetwork(
    backend, weights.joint, config.vocabSize, durations.length
  );
  return new TDTGreedyDecoder(backend, predNet, jointNet, durations, blankId);
}