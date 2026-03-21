// Main application
export { SpeechRecognizer, type SpeechRecognizerConfig, type ASRResult }
  from '../applications/speech/SpeechRecognizer';

// Encoder
export { FastConformerEncoder, type StreamingEncoderState }
  from './encoder/FastConformerEncoder';

// Decoders
export { RNNTGreedyDecoder } from './decoder/RNNTGreedyDecoder';
export { TDTGreedyDecoder } from './decoder/TDTGreedyDecoder';
export { createDecoder, type GreedyDecoder } from './decoder/createDecoder';
export { PredictionNetwork, type PredictionState } from './decoder/PredictionNetwork';
export { RNNTJointNetwork } from './decoder/RNNTJointNetwork';
export { TDTJointNetwork } from './decoder/TDTJointNetwork';

// Features
export { FeaturePipeline } from './features/FeaturePipeline';
export { Resampler } from './features/Resampler';

// Text
export { SentencePieceDecoder } from './text/SentencePieceDecoder';

// Compute
export { TfjsBackend, type TfjsBackendName, type TfjsInitOptions } from './compute/TfjsBackend';
export type { ComputeBackend } from './compute/Backend';
export { ComputeScope } from './compute/ComputeScope';
export type { TensorHandle, Shape, Dtype } from './compute/types';

// Streaming
export { ChunkedInference, type StreamingResult, type ChunkedInferenceConfig }
  from './streaming/ChunkedInference';
export { CacheManager, type StreamingCache } from './streaming/CacheManager';
export { Endpointer, type EndpointerConfig, type EndpointEvent }
  from './streaming/Endpointer';

// Model
export { loadSafeTensors, loadSafeTensorsFromBuffer } from './model/SafeTensorsLoader';
export { parseModelConfig, type FastConformerConfig, type DecoderType } from './model/ModelConfig';
export { mapWeights, type ModelWeights, type EncoderWeights, type DecoderWeights }
  from './model/WeightMapper';
