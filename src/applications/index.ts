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
export { BaseApplication, type ApplicationConfig } from './base/BaseApplication';
export type {
  VADResult,
  PitchResult,
  FormantResult,
  BeepDetectionResult,
  DenoisedFrame,
  FeatureVector
} from './base/types';

// Speech applications
export { VAD, type VADConfig } from './speech/VAD';
export { SpeechRecognizer, type SpeechRecognizerConfig, type ASRResult } from './speech/SpeechRecognizer';

// Detection applications
export { VoicemailBeepDetector, type VoicemailBeepDetectorConfig, type FrequencyRange } from './detection/VoicemailBeepDetector';

// Processing applications
export { AudioDenoiser, type AudioDenoiserConfig } from './processing/AudioDenoiser';
