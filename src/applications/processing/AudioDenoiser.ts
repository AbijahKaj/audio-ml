/**
 * Audio Denoiser
 * Removes noise from audio using spectral subtraction and adaptive noise estimation
 */

import { BaseApplication, type ApplicationConfig } from '../base/BaseApplication.js';
import { type DenoisedFrame } from '../base/types.js';
import FFT from 'fft.js';
import { RMSEAnalyzer } from '../../analysis/RMSEAnalyzer.js';
import { SpectralFlatnessAnalyzer } from '../../analysis/SpectralFlatnessAnalyzer.js';

export interface AudioDenoiserConfig extends ApplicationConfig {
  /** FFT size (default: 2048) */
  fftSize?: number;
  /** Over-subtraction factor (default: 2.0, higher = more aggressive) */
  overSubtractionFactor?: number;
  /** Spectral floor (default: 0.02, prevents over-suppression) */
  spectralFloor?: number;
  /** Number of frames to use for noise estimation (default: 10) */
  noiseEstimationFrames?: number;
  /** Energy threshold for noise estimation (default: 0.01) */
  noiseEstimationThreshold?: number;
  /** Enable adaptive noise tracking (default: true) */
  adaptiveTracking?: boolean;
}

export class AudioDenoiser extends BaseApplication {
  private fft: FFT;
  private fftSize: number;
  private rmse: RMSEAnalyzer;
  private spectralFlatness: SpectralFlatnessAnalyzer;

  private overSubtractionFactor: number;
  private spectralFloor: number;
  private adaptiveTracking: boolean;

  private noiseSpectrum: Float32Array | null = null;
  private noiseEstimationBuffer: Float32Array[] = [];
  private noiseEstimationFrames: number;
  private noiseEstimationThreshold: number;
  private frameCount: number = 0;
  private isNoiseEstimationComplete: boolean = false;

  private window: Float32Array;

  constructor(config: AudioDenoiserConfig) {
    super(config);
    
    this.fftSize = config.fftSize || 2048;
    this.overSubtractionFactor = config.overSubtractionFactor || 2.0;
    this.spectralFloor = config.spectralFloor || 0.02;
    this.adaptiveTracking = config.adaptiveTracking !== false;
    this.noiseEstimationFrames = config.noiseEstimationFrames || 10;
    this.noiseEstimationThreshold = config.noiseEstimationThreshold || 0.01;

    this.fft = new FFT(this.fftSize);
    this.rmse = new RMSEAnalyzer({ sampleRate: config.sampleRate });
    this.spectralFlatness = new SpectralFlatnessAnalyzer({ 
      sampleRate: config.sampleRate, 
      fftSize: this.fftSize 
    });

    // Create Hann window to reduce spectral leakage
    this.window = new Float32Array(this.fftSize);
    for (let i = 0; i < this.fftSize; i++) {
      this.window[i] = 0.5 * (1 - Math.cos(2 * Math.PI * i / (this.fftSize - 1)));
    }
  }

  processFrame(pcm: Float32Array): DenoisedFrame {
    this.frameCount++;

    // Pad or truncate to fftSize
    const padded = new Float32Array(this.fftSize);
    const copyLength = Math.min(pcm.length, this.fftSize);
    padded.set(pcm.subarray(0, copyLength), 0);

    // Apply window
    const windowed = new Float32Array(this.fftSize);
    for (let i = 0; i < this.fftSize; i++) {
      windowed[i] = padded[i] * this.window[i];
    }

    // Compute FFT
    const spectrum = this.fft.createComplexArray();
    this.fft.realTransform(spectrum, windowed);
    this.fft.completeSpectrum(spectrum);

    // Extract magnitude and phase
    const magnitudes = new Float32Array(this.fftSize / 2);
    const phases = new Float32Array(this.fftSize / 2);
    for (let i = 0; i < this.fftSize / 2; i++) {
      const re = spectrum[2 * i];
      const im = spectrum[2 * i + 1];
      magnitudes[i] = Math.sqrt(re * re + im * im);
      phases[i] = Math.atan2(im, re);
    }

    // Estimate noise if not done yet
    if (!this.isNoiseEstimationComplete) {
      // Use windowed frame for analysis (better for spectral features)
      const rmse = this.rmse.analyzeFrame(windowed);
      const flatness = this.spectralFlatness.analyzeFrame(windowed);

      // Consider it noise if low energy and high flatness
      // Made thresholds more lenient: 3x energy threshold, lower flatness requirement (0.4)
      // This makes it easier to detect background noise
      const isNoiseFrame = rmse < this.noiseEstimationThreshold * 3 && flatness > 0.4;
      
      if (isNoiseFrame) {
        this.noiseEstimationBuffer.push(new Float32Array(magnitudes));
        
        // Emit progress update every frame
        const progress = Math.min(100, (this.noiseEstimationBuffer.length / this.noiseEstimationFrames) * 100);
        this.emit('noise-estimation-progress', { progress, frames: this.noiseEstimationBuffer.length });
        
        if (this.noiseEstimationBuffer.length >= this.noiseEstimationFrames) {
          // Average noise spectrum
          this.noiseSpectrum = new Float32Array(this.fftSize / 2);
          for (let i = 0; i < this.fftSize / 2; i++) {
            let sum = 0;
            for (const buffer of this.noiseEstimationBuffer) {
              sum += buffer[i];
            }
            this.noiseSpectrum[i] = sum / this.noiseEstimationBuffer.length;
          }
          this.isNoiseEstimationComplete = true;
          this.emit('noise-estimated', { spectrum: this.noiseSpectrum });
        }
      } else {
        // Only reset buffer if we have very few frames (less than 2)
        // This allows for occasional non-noise frames without losing progress
        if (this.noiseEstimationBuffer.length < 2) {
          this.noiseEstimationBuffer = [];
        }
        // Otherwise keep the buffer and continue - we'll accept frames that are "close enough"
      }
    }

    // Apply denoising if noise estimate is available
    if (this.noiseSpectrum && this.isNoiseEstimationComplete) {
      // Spectral subtraction
      const denoisedMagnitudes = new Float32Array(this.fftSize / 2);
      let noiseReduction = 0;
      let originalEnergy = 0;
      let denoisedEnergy = 0;

      for (let i = 0; i < this.fftSize / 2; i++) {
        originalEnergy += magnitudes[i] * magnitudes[i];
        
        // Subtract noise with over-subtraction
        const subtracted = magnitudes[i] - (this.overSubtractionFactor * this.noiseSpectrum[i]);
        
        // Apply spectral floor to prevent over-suppression
        denoisedMagnitudes[i] = Math.max(
          subtracted,
          this.spectralFloor * magnitudes[i]
        );
        
        denoisedEnergy += denoisedMagnitudes[i] * denoisedMagnitudes[i];
      }

      noiseReduction = 1 - (denoisedEnergy / (originalEnergy + 1e-12));

      // Reconstruct spectrum with original phase
      const denoisedSpectrum = this.fft.createComplexArray();
      for (let i = 0; i < this.fftSize / 2; i++) {
        const re = denoisedMagnitudes[i] * Math.cos(phases[i]);
        const im = denoisedMagnitudes[i] * Math.sin(phases[i]);
        denoisedSpectrum[2 * i] = re;
        denoisedSpectrum[2 * i + 1] = im;
      }
      
      // Mirror for negative frequencies
      for (let i = 1; i < this.fftSize / 2; i++) {
        denoisedSpectrum[2 * (this.fftSize - i)] = denoisedSpectrum[2 * i];
        denoisedSpectrum[2 * (this.fftSize - i) + 1] = -denoisedSpectrum[2 * i + 1];
      }

      // Inverse FFT
      const denoisedTime = new Float32Array(this.fftSize);
      this.fft.inverseTransform(denoisedTime, denoisedSpectrum);
      
      // Normalize (synthesis window is NOT applied — the analysis window already
      // shaped the spectrum; applying it again would double-attenuate the signal)
      for (let i = 0; i < this.fftSize; i++) {
        denoisedTime[i] /= this.fftSize;
      }

      // Calculate SNR (approximate)
      const signalEnergy = Array.from(magnitudes).reduce((sum, m) => sum + m * m, 0);
      const noiseEnergy = Array.from(this.noiseSpectrum).reduce((sum, n) => sum + n * n, 0);
      const snr = 10 * Math.log10((signalEnergy + 1e-12) / (noiseEnergy + 1e-12));

      // Update noise estimate adaptively if enabled
      if (this.adaptiveTracking) {
        const rmse = this.rmse.analyzeFrame(pcm);
        const flatness = this.spectralFlatness.analyzeFrame(pcm);
        
        // Update noise estimate during silence
        if (rmse < this.noiseEstimationThreshold && flatness > 0.7) {
          for (let i = 0; i < this.fftSize / 2; i++) {
            // Exponential moving average
            this.noiseSpectrum[i] = 0.9 * this.noiseSpectrum[i] + 0.1 * magnitudes[i];
          }
        }
      }

      this.emit('snr-updated', { snr });
      this.emit('denoised-frame', { audio: denoisedTime, snr, noiseReduction });

      return {
        audio: denoisedTime,
        snr,
        noiseReduction
      };
    }

    // Return original if noise not estimated yet
    return {
      audio: windowed,
      snr: 0,
      noiseReduction: 0
    };
  }

  reset(): void {
    super.reset();
    this.noiseSpectrum = null;
    this.noiseEstimationBuffer = [];
    this.frameCount = 0;
    this.isNoiseEstimationComplete = false;
  }

  /**
   * Whether noise estimation has completed
   */
  get noiseEstimated(): boolean {
    return this.isNoiseEstimationComplete;
  }

  /**
   * Manually set noise spectrum estimate
   */
  setNoiseEstimate(spectrum: Float32Array): void {
    this.noiseSpectrum = new Float32Array(spectrum);
    this.isNoiseEstimationComplete = true;
    this.emit('noise-estimated', { spectrum: this.noiseSpectrum });
  }
}
