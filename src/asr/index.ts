export {
  SpeechRecognizer,
  type SpeechRecognizerConfig,
  type SpeechRecognizerProvider,
  type TransformersDtype,
  type ASRResult,
} from '../applications/speech/SpeechRecognizer.js';

export { TfjsBackend } from './compute/TfjsBackend.js';
export type { ComputeBackend } from './compute/Backend.js';
export type { BackendKind, Dtype, Shape, TensorHandle } from './compute/types.js';

export { FeaturePipeline } from './features/FeaturePipeline.js';
export { Resampler } from './features/Resampler.js';

export { FastConformerEncoder } from './encoder/FastConformerEncoder.js';
export { ConvSubsampling } from './encoder/ConvSubsampling.js';
export { ConformerBlock } from './encoder/ConformerBlock.js';
export { ConvModule } from './encoder/ConvModule.js';
export { FeedForward } from './encoder/FeedForward.js';
export { Linear } from './encoder/Linear.js';
export { MultiHeadAttention } from './encoder/MultiHeadAttention.js';
export { RelativePositionalEncoding } from './encoder/RelativePositionalEncoding.js';

export { PredictionNetwork } from './decoder/PredictionNetwork.js';
export { JointNetwork } from './decoder/JointNetwork.js';
export { RNNTJointNetwork } from './decoder/RNNTJointNetwork.js';
export { TDTJointNetwork } from './decoder/TDTJointNetwork.js';
export { RNNTGreedyDecoder } from './decoder/RNNTGreedyDecoder.js';
export { TDTGreedyDecoder } from './decoder/TDTGreedyDecoder.js';
export { createDecoder } from './decoder/createDecoder.js';

export { CacheManager, type StreamingCache } from './streaming/CacheManager.js';
export { ChunkedInference } from './streaming/ChunkedInference.js';
export { Endpointer, type EndpointState } from './streaming/Endpointer.js';

export { SentencePieceDecoder } from './text/SentencePieceDecoder.js';

export { parseModelConfig, type DecoderType, type FastConformerConfig } from './model/ModelConfig.js';
export { loadSafeTensors } from './model/SafeTensorsLoader.js';
export { mapWeights } from './model/WeightMapper.js';
