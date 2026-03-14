/**
 * Voicemail Beep Detector
 * Detects voicemail beeps and tones using frequency analysis
 */

import { BaseApplication, type ApplicationConfig } from '../base/BaseApplication';
import { type BeepDetectionResult } from '../base/types';
import { FFTAnalyzer } from '../../analysis/FFTAnalyzer';

export interface FrequencyRange {
  min: number;
  max: number;
  name?: string;
}

export interface VoicemailBeepDetectorConfig extends ApplicationConfig {
  /** FFT size (default: 2048 for better frequency resolution) */
  fftSize?: number;
  /** Minimum beep duration in seconds (default: 0.1) */
  minBeepDuration?: number;
  /** Maximum beep duration in seconds (default: 2.0) */
  maxBeepDuration?: number;
  /** Frequency ranges to detect beeps in (Hz) */
  frequencyRanges?: FrequencyRange[];
  /** Energy threshold for tone detection (default: 0.05) */
  energyThreshold?: number;
  /** Peak prominence threshold - how much stronger the peak must be than surrounding frequencies (default: 2.0) */
  peakProminence?: number;
  /** Minimum peak magnitude relative to total energy (default: 0.3) */
  minPeakMagnitude?: number;
}

export class VoicemailBeepDetector extends BaseApplication {
  private fft: FFTAnalyzer;

  private fftSize: number;
  private minBeepDuration: number;
  private maxBeepDuration: number;
  private frequencyRanges: FrequencyRange[];
  private energyThreshold: number;
  private peakProminence: number;
  private minPeakMagnitude: number;

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
    this.frequencyRanges = config.frequencyRanges || [
      { min: 400, max: 500, name: 'Low beep' },
      { min: 900, max: 1100, name: 'Mid beep' },
      { min: 1400, max: 1600, name: 'High beep' },
      { min: 1900, max: 2100, name: 'Very high beep' }
    ];
    this.energyThreshold = config.energyThreshold || 0.05;
    this.peakProminence = config.peakProminence || 2.0;
    this.minPeakMagnitude = config.minPeakMagnitude || 0.3;

    this.frameDuration = this.fftSize / config.sampleRate;

    this.fft = new FFTAnalyzer({ sampleRate: config.sampleRate, fftSize: this.fftSize });
  }

  processFrame(pcm: Float32Array): BeepDetectionResult | null {
    this.frameCount++;

    // Pad or truncate to fftSize so FFTAnalyzer doesn't throw
    let frame = pcm;
    if (pcm.length !== this.fftSize) {
      frame = new Float32Array(this.fftSize);
      frame.set(pcm.subarray(0, Math.min(pcm.length, this.fftSize)));
    }

    // Get frequency spectrum
    const spectrum = this.fft.analyzeFrame(frame);
    
    // Calculate total energy
    const totalEnergy = Array.from(spectrum).reduce((sum, mag) => sum + mag * mag, 0);
    const avgEnergy = totalEnergy / spectrum.length;

    // Check if we have enough energy overall
    if (avgEnergy < this.energyThreshold) {
      // Not enough energy, check if we should end current tone
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
          this.emit('tone-end', {
            frequency: this.currentTone.frequency,
            duration
          });
          this.currentTone = null;
        } else {
          this.currentTone = null;
        }
      }
      return null;
    }

    // Find peaks in each frequency range
    let detectedFrequency: number | null = null;
    let peakMagnitude = 0;

    for (const range of this.frequencyRanges) {
      // Convert frequency range to bin indices (bin k = k * sampleRate / fftSize)
      const minBin = Math.floor((range.min * this.fftSize) / this.sampleRate);
      const maxBin = Math.ceil((range.max * this.fftSize) / this.sampleRate);
      
      // Find peak in this range
      let rangeMaxMagnitude = 0;
      let rangePeakBin = 0;
      let rangeSum = 0;
      
      for (let bin = minBin; bin < maxBin && bin < spectrum.length; bin++) {
        const magnitude = spectrum[bin];
        rangeSum += magnitude;
        if (magnitude > rangeMaxMagnitude) {
          rangeMaxMagnitude = magnitude;
          rangePeakBin = bin;
        }
      }
      
      const binCount = maxBin - minBin;
      if (binCount <= 0) continue;

      const rangeAvgMagnitude = rangeSum / binCount;
      
      // Check if peak is prominent (stands out from surrounding frequencies)
      if (rangeMaxMagnitude > 0 && rangeAvgMagnitude > 0) {
        const prominence = rangeMaxMagnitude / rangeAvgMagnitude;
        
        // Check if peak is strong enough relative to total energy
        const peakEnergyRatio = rangeMaxMagnitude / (Math.sqrt(totalEnergy) + 1e-10);
        
        if (prominence >= this.peakProminence && peakEnergyRatio >= this.minPeakMagnitude) {
          // This is a valid peak in the range
          if (rangeMaxMagnitude > peakMagnitude) {
            peakMagnitude = rangeMaxMagnitude;
            detectedFrequency = (rangePeakBin * this.sampleRate) / this.fftSize;
          }
        }
      }
    }

    // Check if we have a sustained tone in a valid frequency range
    const isTone = detectedFrequency !== null && peakMagnitude > 0;

    if (isTone && detectedFrequency !== null) {
      if (this.currentTone === null) {
        // Start of new tone
        this.currentTone = {
          frequency: detectedFrequency,
          startTime: this.frameCount * this.frameDuration,
          energy: peakMagnitude
        };
        this.emit('tone-start', { frequency: detectedFrequency, energy: peakMagnitude });
      } else {
        // Continue existing tone - update if frequency is similar (within 100 Hz)
        if (Math.abs(this.currentTone.frequency - detectedFrequency) < 100) {
          this.currentTone.energy = Math.max(this.currentTone.energy, peakMagnitude);
          // Update frequency to track slight variations
          this.currentTone.frequency = detectedFrequency;
        } else {
          // Frequency changed significantly - end old tone and start new one
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
          }
          // Start new tone
          this.currentTone = {
            frequency: detectedFrequency,
            startTime: this.frameCount * this.frameDuration,
            energy: peakMagnitude
          };
          this.emit('tone-start', { frequency: detectedFrequency, energy: peakMagnitude });
        }
      }
    } else {
      // No tone detected - check if we just ended a tone that qualifies as a beep
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
