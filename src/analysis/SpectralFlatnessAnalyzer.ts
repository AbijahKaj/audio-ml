/**
 * SpectralFlatnessAnalyzer computes the spectral flatness of a PCM audio frame.
 * Spectral flatness measures how noise-like a sound is (flat = noise, peaked = tone).
 * Usage: const sf = new SpectralFlatnessAnalyzer({ sampleRate: 16000 }); sf.analyzeFrame(pcm)
 */
import FFT from 'fft.js';

export interface SpectralFlatnessConfig {
  /** Sample rate of the audio (Hz) */
  sampleRate: number;
  /** FFT size (default: 1024) */
  fftSize?: number;
}

export class SpectralFlatnessAnalyzer {
  private fft: FFT;
  private fftSize: number;
  /**
   * @param config - Configuration with sample rate and optional FFT size
   */
  constructor(config: SpectralFlatnessConfig) {
    this.fftSize = config.fftSize || 1024;
    this.fft = new FFT(this.fftSize);
  }
  /**
   * Compute the spectral flatness for a PCM frame.
   * @param pcm - Input PCM audio frame
   * @returns Spectral flatness (0 = tone, 1 = noise)
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
    const geoMean = Math.exp(mags.reduce((a, b) => a + Math.log(b + 1e-12), 0) / mags.length);
    const arithMean = mags.reduce((a, b) => a + b, 0) / mags.length;
    return geoMean / (arithMean + 1e-12);
  }
}
