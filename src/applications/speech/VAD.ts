/**
 * Voice Activity Detection (VAD)
 * Detects when speech is present vs silence/noise using multiple audio features
 */

import { BaseApplication, type ApplicationConfig } from '../base/BaseApplication';
import { type VADResult } from '../base/types';
import { RMSEAnalyzer } from '../../analysis/RMSEAnalyzer';
import { ZeroCrossingRateAnalyzer } from '../../analysis/ZeroCrossingRateAnalyzer';
import { SpectralFlatnessAnalyzer } from '../../analysis/SpectralFlatnessAnalyzer';
import { SpectralCentroidAnalyzer } from '../../analysis/SpectralCentroidAnalyzer';

export interface VADConfig extends ApplicationConfig {
  /** Energy threshold for speech detection (default: 0.01) */
  energyThreshold?: number;
  /** ZCR threshold for voiced/unvoiced detection (default: 0.1) */
  zcrThreshold?: number;
  /** Spectral flatness threshold (lower = more speech-like, default: 0.7) */
  flatnessThreshold?: number;
  /** Minimum spectral centroid for speech (Hz, default: 500) */
  minCentroid?: number;
  /** Number of consecutive speech frames required to trigger speech-start (default: 3) */
  speechFramesRequired?: number;
  /** Number of consecutive silence frames required to trigger speech-end (default: 5) */
  silenceFramesRequired?: number;
  /** FFT size (default: 1024) */
  fftSize?: number;
}

export class VAD extends BaseApplication {
  private rmse: RMSEAnalyzer;
  private zcr: ZeroCrossingRateAnalyzer;
  private spectralFlatness: SpectralFlatnessAnalyzer;
  private spectralCentroid: SpectralCentroidAnalyzer;

  private energyThreshold: number;
  private zcrThreshold: number;
  private flatnessThreshold: number;
  private minCentroid: number;
  private speechFramesRequired: number;
  private silenceFramesRequired: number;

  private consecutiveSpeechFrames: number = 0;
  private consecutiveSilenceFrames: number = 0;
  private isCurrentlySpeech: boolean = false;

  constructor(config: VADConfig) {
    super(config);
    
    this.energyThreshold = config.energyThreshold ?? 0.01;
    this.zcrThreshold = config.zcrThreshold ?? 0.1;
    this.flatnessThreshold = config.flatnessThreshold ?? 0.7;
    this.minCentroid = config.minCentroid ?? 500;
    this.speechFramesRequired = config.speechFramesRequired ?? 3;
    this.silenceFramesRequired = config.silenceFramesRequired ?? 5;

    const fftSize = config.fftSize || 1024;

    this.rmse = new RMSEAnalyzer({ sampleRate: config.sampleRate });
    this.zcr = new ZeroCrossingRateAnalyzer({ sampleRate: config.sampleRate });
    this.spectralFlatness = new SpectralFlatnessAnalyzer({ 
      sampleRate: config.sampleRate, 
      fftSize 
    });
    this.spectralCentroid = new SpectralCentroidAnalyzer({ 
      sampleRate: config.sampleRate, 
      fftSize 
    });
  }

  processFrame(pcm: Float32Array): VADResult {
    // Extract features
    const rmse = this.rmse.analyzeFrame(pcm);
    const zcr = this.zcr.analyzeFrame(pcm);
    const flatness = this.spectralFlatness.analyzeFrame(pcm);
    const centroid = this.spectralCentroid.analyzeFrame(pcm);

    // Determine if frame is speech-like
    const hasEnergy = rmse > this.energyThreshold;
    const hasLowZCR = zcr < this.zcrThreshold; // Voiced speech has lower ZCR
    const hasLowFlatness = flatness < this.flatnessThreshold; // Speech is less flat than noise
    const hasCentroid = centroid > this.minCentroid; // Speech has higher frequency content

    // Combine features with weighted decision
    // Require multiple indicators for speech, not just energy
    const energyScore = hasEnergy ? 0.4 : 0; // Reduced weight - energy alone shouldn't determine speech
    const zcrScore = hasLowZCR ? 0.3 : 0;
    const flatnessScore = hasLowFlatness ? 0.2 : 0;
    const centroidScore = hasCentroid ? 0.1 : 0;

    const confidence = Math.min(1.0, energyScore + zcrScore + flatnessScore + centroidScore);
    // Require at least 2 indicators (energy + one other) for speech
    const isSpeechFrame = confidence >= 0.5 && (hasEnergy && (hasLowZCR || hasLowFlatness || hasCentroid));

    // Update state with temporal smoothing
    if (isSpeechFrame) {
      this.consecutiveSpeechFrames++;
      this.consecutiveSilenceFrames = 0;
    } else {
      this.consecutiveSilenceFrames++;
      this.consecutiveSpeechFrames = 0;
    }

    // Emit events based on state changes
    const wasSpeech = this.isCurrentlySpeech;
    
    if (!wasSpeech && this.consecutiveSpeechFrames >= this.speechFramesRequired) {
      this.isCurrentlySpeech = true;
      this.emit('speech-start', { confidence, features: { rmse, zcr, flatness, centroid } });
    } else if (wasSpeech && this.consecutiveSilenceFrames >= this.silenceFramesRequired) {
      this.isCurrentlySpeech = false;
      this.emit('speech-end', { confidence, features: { rmse, zcr, flatness, centroid } });
    }

    // Emit continuous frame updates
    this.emit('frame', {
      isSpeech: this.isCurrentlySpeech,
      confidence,
      features: { rmse, zcr, flatness, centroid }
    });

    return {
      isSpeech: this.isCurrentlySpeech,
      confidence,
      features: { rmse, zcr, spectralFlatness: flatness, spectralCentroid: centroid }
    };
  }

  reset(): void {
    super.reset();
    this.consecutiveSpeechFrames = 0;
    this.consecutiveSilenceFrames = 0;
    this.isCurrentlySpeech = false;
  }

  /**
   * Get current VAD state
   */
  getState(): { isSpeech: boolean; confidence: number } {
    return {
      isSpeech: this.isCurrentlySpeech,
      confidence: this.isCurrentlySpeech ? 0.8 : 0.2
    };
  }
}
