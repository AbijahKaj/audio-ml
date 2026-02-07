/**
 * FFTAnalyzer computes the Fast Fourier Transform magnitude spectrum of a PCM audio frame.
 * This shows the frequency content of the signal as a magnitude spectrum.
 * Usage: const fft = new FFTAnalyzer({ sampleRate: 16000 }); fft.analyzeFrame(pcm)
 */
import FFT from 'fft.js';

export interface FFTConfig {
  sampleRate: number;
  fftSize?: number;
}

export class FFTAnalyzer {
  private fft: FFT;
  private fftSize: number;

  constructor(config: FFTConfig) {
    this.fftSize = config.fftSize || 1024;
    this.fft = new FFT(this.fftSize);
  }

  /**
   * Compute the FFT magnitude spectrum for a PCM frame.
   * @param pcm - Input PCM audio frame (must be fftSize samples)
   * @returns Array of magnitude values (one per frequency bin)
   */
  analyzeFrame(pcm: Float32Array): Float32Array {
    if (pcm.length !== this.fftSize) {
      throw new Error(`PCM frame must be fftSize (${this.fftSize}) samples`);
    }
    
    const spectrum = this.fft.createComplexArray();
    this.fft.realTransform(spectrum, pcm);
    this.fft.completeSpectrum(spectrum);
    
    // Compute magnitude spectrum (only need first half for real signals)
    const magnitudes = new Float32Array(this.fftSize / 2);
    for (let i = 0; i < this.fftSize / 2; i++) {
      const re = spectrum[2 * i];
      const im = spectrum[2 * i + 1];
      magnitudes[i] = Math.sqrt(re * re + im * im);
    }
    
    return magnitudes;
  }
}
