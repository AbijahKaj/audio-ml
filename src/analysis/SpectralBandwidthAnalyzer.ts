/**
 * SpectralBandwidthAnalyzer computes the spectral bandwidth of a PCM audio frame.
 * The spectral bandwidth is the spread of the spectrum around the centroid, related to the perceived timbre of a sound.
 * Usage: const sb = new SpectralBandwidthAnalyzer({ sampleRate: 16000 }); sb.analyzeFrame(pcm)
 */
import FFT from 'fft.js';

export interface SpectralBandwidthConfig {
  /** Sample rate of the audio (Hz) */
  sampleRate: number;
  /** FFT size (default: 1024) */
  fftSize?: number;
}

export class SpectralBandwidthAnalyzer {
  private fft: FFT;
  private fftSize: number;
  private sampleRate: number;
  /**
   * @param config - Configuration with sample rate and optional FFT size
   */
  constructor(config: SpectralBandwidthConfig) {
    this.fftSize = config.fftSize || 1024;
    this.fft = new FFT(this.fftSize);
    this.sampleRate = config.sampleRate;
  }
  /**
   * Compute the spectral bandwidth for a PCM frame.
   * @param pcm - Input PCM audio frame
   * @returns Spectral bandwidth in Hz
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
    // Spectral centroid
    let num = 0, denom = 0;
    for (let i = 0; i < mags.length; i++) {
      num += i * mags[i];
      denom += mags[i];
    }
    if (denom === 0) return 0;
    const centroid = (num / denom);
    // Bandwidth
    let spread = 0;
    for (let i = 0; i < mags.length; i++) {
      spread += mags[i] * Math.pow(i - centroid, 2);
    }
    return Math.sqrt(spread / denom) * (this.sampleRate / 2) / mags.length;
  }
}
