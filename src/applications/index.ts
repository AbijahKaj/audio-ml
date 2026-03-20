/**
 * Audio Applications
 * 
 * High-level applications built on top of audio analyzers.
 * All applications use EventEmitter for real-time event handling.
 * 
 * @example
 * ```typescript
 * import { VAD, AudioDenoiser } from 'audio-ml/applications';
 * 
 * const vad = new VAD({ sampleRate: 44100 });
 * vad.on('speech-start', () => console.log('Speech detected!'));
 * const result = vad.processFrame(pcmData);
 * ```
 */

// Base classes and types
export { BaseApplication, type ApplicationConfig } from './base/BaseApplication.js';
export type {
  VADResult,
  PitchResult,
  FormantResult,
  BeepDetectionResult,
  DenoisedFrame,
  FeatureVector
} from './base/types.js';

// Speech applications
export { VAD, type VADConfig } from './speech/VAD.js';
export {
  SpeechRecognizer,
  type SpeechRecognizerConfig,
  type SpeechRecognizerProvider,
  type TransformersDtype,
  type ASRResult,
} from './speech/SpeechRecognizer.js';

// Detection applications
export { VoicemailBeepDetector, type VoicemailBeepDetectorConfig, type FrequencyRange } from './detection/VoicemailBeepDetector.js';

// Processing applications
export { AudioDenoiser, type AudioDenoiserConfig } from './processing/AudioDenoiser.js';
