/**
 * audio-ml/asr — FastConformer ASR Engine
 *
 * Supports RNNT and TDT decoder variants with NeMo-compatible weights.
 * Works in browsers (WASM/WebGPU) and Node.js.
 *
 * Quick start:
 * ```typescript
 * import { SpeechRecognizer } from 'audio-ml/asr';
 *
 * const recognizer = new SpeechRecognizer({
 *   sampleRate: 16000,
 *   modelPath: './parakeet_120m.safetensors',
 *   configPath: './model_config.json',
 *   vocabPath: './vocab.json',
 *   backend: 'wasm',
 * });
 *
 * await recognizer.load();
 * const { text } = await recognizer.transcribe(audioFloat32Array);
 * ```
 */

// Main application
export {
  SpeechRecognizer,
  type SpeechRecognizerConfig,
  type ASRResult,
  type ASRPartialResult,
  type ASRFinalResult,
} from '../applications/speech/SpeechRecognizer.js';

// Compute backend
export { TfjsBackend } from './compute/TfjsBackend.js';
export { ComputeScope } from './compute/ComputeScope.js';
export type { ComputeBackend } from './compute/Backend.js';
export type { TensorHandle, Shape, Dtype } from './compute/types.js';

// Model loading
export { parseModelConfig, type FastConformerConfig, type DecoderType } from './model/ModelConfig.js';
export { loadSafeTensors } from './model/SafeTensorsLoader.js';
export {
  mapWeights,
  type ModelWeights,
  type EncoderWeights,
  type DecoderWeights,
} from './model/WeightMapper.js';

// Encoder
export { FastConformerEncoder, type EncoderCache } from './encoder/FastConformerEncoder.js';

// Decoders
export { RNNTGreedyDecoder } from './decoder/RNNTGreedyDecoder.js';
export { TDTGreedyDecoder } from './decoder/TDTGreedyDecoder.js';
export { createDecoder } from './decoder/createDecoder.js';
export { PredictionNetwork, type LSTMState } from './decoder/PredictionNetwork.js';

// Feature pipeline
export { FeaturePipeline } from './features/FeaturePipeline.js';
export { Resampler } from './features/Resampler.js';

// Text
export { SentencePieceDecoder } from './text/SentencePieceDecoder.js';

// Streaming
export { CacheManager, type StreamingCache } from './streaming/CacheManager.js';
export { ChunkedInference, type ChunkResult } from './streaming/ChunkedInference.js';
export { Endpointer, type EndpointEvent, type EndpointerConfig } from './streaming/Endpointer.js';
