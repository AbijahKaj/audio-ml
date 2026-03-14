/**
 * Shared types for applications
 */

export interface VADResult {
  isSpeech: boolean;
  confidence: number;
  features: {
    rmse: number;
    zcr: number;
    spectralFlatness: number;
    spectralCentroid: number;
  };
}

export interface PitchResult {
  pitch: number; // Hz
  confidence: number;
  voiced: boolean;
}

export interface FormantResult {
  f1: number; // First formant (Hz)
  f2: number; // Second formant (Hz)
  f3: number; // Third formant (Hz)
  confidence: number;
}

export interface BeepDetectionResult {
  frequency: number;
  duration: number;
  type: 'beep' | 'tone';
  confidence: number;
}

export interface DenoisedFrame {
  audio: Float32Array;
  snr: number;
  noiseReduction: number;
}

export interface FeatureVector {
  mfcc: number[];
  spectral: {
    centroid: number;
    rolloff: number;
    bandwidth: number;
    flatness: number;
  };
  energy: number;
  zcr: number;
}
