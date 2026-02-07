/**
 * Voicemail Beep Detector
 * Detects voicemail beeps and tones using frequency analysis
 */

import { BaseApplication, type ApplicationConfig } from '../base/BaseApplication';
import { type BeepDetectionResult } from '../base/types';
import { FFTAnalyzer } from '../../analysis/FFTAnalyzer';
import { SpectralCentroidAnalyzer } from '../../analysis/SpectralCentroidAnalyzer';

export interface VoicemailBeepDetectorConfig extends ApplicationConfig {
  /** FFT size (default: 2048 for better frequency resolution) */
  fftSize?: number;
  /** Minimum beep duration in seconds (default: 0.1) */
  minBeepDuration?: number;
  /** Maximum beep duration in seconds (default: 2.0) */
  maxBeepDuration?: number;
  /** Frequency tolerance in Hz (default: 50) */
  frequencyTolerance?: number;
  /** Common beep frequencies to detect (Hz) */
  beepFrequencies?: number[];
  /** Energy threshold for tone detection (default: 0.05) */
  energyThreshold?: number;
}

export class VoicemailBeepDetector extends BaseApplication {
  private fft: FFTAnalyzer;
  private spectralCentroid: SpectralCentroidAnalyzer;

  private fftSize: number;
  private minBeepDuration: number;
  private maxBeepDuration: number;
  private frequencyTolerance: number;
  private beepFrequencies: number[];
  private energyThreshold: number;

  private currentTone: {
    frequency: number;
    startTime: number;
    energy: number;
  } | null = null;
  private frameCount: number = 0;
  private frameDuration: number;

  constructor(config: VoicemailBeepDetectorConfig) {
    super(config);
    
    this.fftSize = config.fftSize || 2048;
    this.minBeepDuration = config.minBeepDuration || 0.1;
    this.maxBeepDuration = config.maxBeepDuration || 2.0;
    this.frequencyTolerance = config.frequencyTolerance || 50;
    this.beepFrequencies = config.beepFrequencies || [440, 1000, 1500, 2000];
    this.energyThreshold = config.energyThreshold || 0.05;

    this.frameDuration = this.fftSize / config.sampleRate;

    this.fft = new FFTAnalyzer({ sampleRate: config.sampleRate, fftSize: this.fftSize });
    this.spectralCentroid = new SpectralCentroidAnalyzer({ 
      sampleRate: config.sampleRate, 
      fftSize: this.fftSize 
    });
  }

  processFrame(pcm: Float32Array): BeepDetectionResult | null {
    this.frameCount++;

    // Get frequency spectrum
    const spectrum = this.fft.analyzeFrame(pcm);
    const centroid = this.spectralCentroid.analyzeFrame(pcm);
    
    // Calculate total energy
    const energy = Array.from(spectrum).reduce((sum, mag) => sum + mag * mag, 0) / spectrum.length;

    // Find dominant frequency
    let maxMagnitude = 0;
    let dominantBin = 0;
    for (let i = 0; i < spectrum.length; i++) {
      if (spectrum[i] > maxMagnitude) {
        maxMagnitude = spectrum[i];
        dominantBin = i;
      }
    }
    const dominantFreq = (dominantBin * this.sampleRate) / (this.fftSize * 2);

    // Check if this matches a known beep frequency
    const matchedFreq = this.beepFrequencies.find(freq => 
      Math.abs(dominantFreq - freq) < this.frequencyTolerance
    );

    // Check if we have a sustained tone
    const isTone = energy > this.energyThreshold && 
                   matchedFreq !== undefined &&
                   Math.abs(centroid - dominantFreq) < this.frequencyTolerance * 2;

    if (isTone) {
      if (this.currentTone === null) {
        // Start of new tone
        this.currentTone = {
          frequency: matchedFreq!,
          startTime: this.frameCount * this.frameDuration,
          energy
        };
        this.emit('tone-start', { frequency: matchedFreq!, energy });
      } else {
        // Continue existing tone
        this.currentTone.energy = Math.max(this.currentTone.energy, energy);
      }
    } else {
      // Check if we just ended a tone that qualifies as a beep
      if (this.currentTone !== null) {
        const duration = (this.frameCount * this.frameDuration) - this.currentTone.startTime;
        
        if (duration >= this.minBeepDuration && duration <= this.maxBeepDuration) {
          const result: BeepDetectionResult = {
            frequency: this.currentTone.frequency,
            duration,
            type: 'beep',
            confidence: Math.min(1.0, this.currentTone.energy / this.energyThreshold)
          };
          
          this.emit('beep-detected', result);
          this.currentTone = null;
          return result;
        } else if (duration > this.maxBeepDuration) {
          // Too long, emit as tone-end
          this.emit('tone-end', {
            frequency: this.currentTone.frequency,
            duration
          });
          this.currentTone = null;
        } else {
          // Too short, ignore
          this.currentTone = null;
        }
      }
    }

    return null;
  }

  reset(): void {
    super.reset();
    this.currentTone = null;
    this.frameCount = 0;
  }
}
