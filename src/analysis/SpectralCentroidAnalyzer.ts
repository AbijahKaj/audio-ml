/**
 * SpectralCentroidAnalyzer computes the spectral centroid of a PCM audio frame.
 * The spectral centroid is the "center of mass" of the spectrum and is related to the perceived brightness of a sound.
 * Usage: const sc = new SpectralCentroidAnalyzer({ sampleRate: 16000 }); sc.analyzeFrame(pcm)
 */
import FFT from 'fft.js';

export interface SpectralCentroidConfig {
  /** Sample rate of the audio (Hz) */
  sampleRate: number;
  /** FFT size (default: 1024) */
  fftSize?: number;
}

export class SpectralCentroidAnalyzer {
  private fft: FFT;
  private fftSize: number;
  private sampleRate: number;
  /**
   * @param config - Configuration with sample rate and optional FFT size
   */
  constructor(config: SpectralCentroidConfig) {
    this.fftSize = config.fftSize || 1024;
    this.fft = new FFT(this.fftSize);
    this.sampleRate = config.sampleRate;
  }
  /**
   * Compute the spectral centroid for a PCM frame.
   * @param pcm - Input PCM audio frame
   * @returns Spectral centroid in Hz
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
    let num = 0, denom = 0;
    for (let i = 0; i < mags.length; i++) {
      num += i * mags[i];
      denom += mags[i];
    }
    if (denom === 0) return 0;
    return (num / denom) * (this.sampleRate / 2) / mags.length;
  }
}
