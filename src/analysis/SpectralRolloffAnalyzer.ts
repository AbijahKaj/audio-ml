/**
 * SpectralRolloffAnalyzer computes the spectral rolloff point of a PCM audio frame.
 * The rolloff is the frequency below which a certain percentage (e.g., 85%) of the total spectral energy lies.
 * Usage: const sr = new SpectralRolloffAnalyzer({ sampleRate: 16000 }); sr.analyzeFrame(pcm)
 */
import FFT from 'fft.js';

export interface SpectralRolloffConfig {
  /** Sample rate of the audio (Hz) */
  sampleRate: number;
  /** FFT size (default: 1024) */
  fftSize?: number;
  /** Rolloff percent (default: 0.85) */
  rolloffPercent?: number;
}

export class SpectralRolloffAnalyzer {
  private fft: FFT;
  private fftSize: number;
  private sampleRate: number;
  private rolloffPercent: number;
  /**
   * @param config - Configuration with sample rate, optional FFT size, and rolloff percent
   */
  constructor(config: SpectralRolloffConfig) {
    this.fftSize = config.fftSize || 1024;
    this.fft = new FFT(this.fftSize);
    this.sampleRate = config.sampleRate;
    this.rolloffPercent = config.rolloffPercent || 0.85;
  }
  /**
   * Compute the spectral rolloff for a PCM frame.
   * @param pcm - Input PCM audio frame
   * @returns Spectral rolloff frequency in Hz
   */
  analyzeFrame(pcm: Float32Array): number {
    const spectrum = this.fft.createComplexArray();
    this.fft.realTransform(spectrum, pcm);
    this.fft.completeSpectrum(spectrum);
    let mags: number[] = [];
    for (let i = 0; i < this.fftSize / 2; i++) {
      const re = spectrum[2 * i];
      const im = spectrum[2 * i + 1];
      mags.push(Math.sqrt(re * re + im * im));
    }
    const totalEnergy = mags.reduce((a, b) => a + b, 0);
    let threshold = totalEnergy * this.rolloffPercent;
    let cumulative = 0;
    for (let i = 0; i < mags.length; i++) {
      cumulative += mags[i];
      if (cumulative >= threshold) {
        return (i / mags.length) * (this.sampleRate / 2);
      }
    }
    return 0;
  }
}
