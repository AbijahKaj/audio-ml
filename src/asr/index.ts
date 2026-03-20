export { SpeechRecognizer, type SpeechRecognizerConfig, type ASRResult, type StreamingASRResult } from '../applications/speech/SpeechRecognizer';

export { TfjsBackend } from './compute/TfjsBackend';
export { ComputeScope } from './compute/ComputeScope';
export type { ComputeBackend } from './compute/Backend';
export type { Dtype, Shape, TensorHandle } from './compute/types';

export { parseModelConfig, type DecoderType, type FastConformerConfig } from './model/ModelConfig';
export { loadSafeTensors } from './model/SafeTensorsLoader';
export { mapWeights, summarizeModelWeights, type WeightStudySummary } from './model/WeightMapper';
export type { ModelWeights, EncoderWeights, DecoderWeights } from './model/ModelWeights';

export { FeaturePipeline } from './features/FeaturePipeline';
export { Resampler } from './features/Resampler';

export { FastConformerEncoder } from './encoder/FastConformerEncoder';

export { PredictionNetwork } from './decoder/PredictionNetwork';
export { JointNetwork } from './decoder/JointNetwork';
export { RNNTJointNetwork } from './decoder/RNNTJointNetwork';
export { TDTJointNetwork } from './decoder/TDTJointNetwork';
export { TransducerDecoder } from './decoder/TransducerDecoder';
export { RNNTGreedyDecoder } from './decoder/RNNTGreedyDecoder';
export { TDTGreedyDecoder } from './decoder/TDTGreedyDecoder';
export { createDecoder } from './decoder/createDecoder';
export type { DecoderState, DecodeResult } from './decoder/types';

export { SentencePieceDecoder } from './text/SentencePieceDecoder';

export { CacheManager, type StreamingCache } from './streaming/CacheManager';
export { ChunkedInference, type ChunkedInferenceResult } from './streaming/ChunkedInference';
export { Endpointer, type EndpointerConfig, type EndpointDecision } from './streaming/Endpointer';
