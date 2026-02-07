/**
 * MelSpectrogramAnalyzer computes mel spectrogram features from PCM audio frames.
 * Mel spectrogram applies a mel filter bank to the FFT spectrum and returns log-mel energies.
 * This is similar to MFCC but without the DCT step.
 * Usage: const mel = new MelSpectrogramAnalyzer({ sampleRate: 16000 }); mel.analyzeFrame(pcm)
 */
import FFT from 'fft.js';

export interface MelSpectrogramConfig {
  sampleRate: number;
  fftSize?: number;
  melBands?: number;
}

export class MelSpectrogramAnalyzer {
  private fft: FFT;
  private fftSize: number;
  private melBands: number;
  private sampleRate: number;
  private melFilters: number[][];

  constructor(config: MelSpectrogramConfig) {
    this.fftSize = config.fftSize || 1024;
    this.melBands = config.melBands || 40;
    this.sampleRate = config.sampleRate;
    this.fft = new FFT(this.fftSize);
    this.melFilters = this.createMelFilterBank();
  }

  /**
   * Analyze a single frame of PCM samples and return mel spectrogram features.
   * @param pcm - Input PCM audio frame (must be fftSize samples)
   * @returns Array of log-mel energies
   */
  analyzeFrame(pcm: Float32Array): number[] {
    if (pcm.length !== this.fftSize) {
      throw new Error(`PCM frame must be fftSize (${this.fftSize}) samples`);
    }
    const spectrum = this.fft.createComplexArray();
    this.fft.realTransform(spectrum, pcm);
    this.fft.completeSpectrum(spectrum);

    // Get magnitude spectrum
    const mags: number[] = [];
    for (let i = 0; i < this.fftSize / 2; i++) {
      const re = spectrum[2 * i];
      const im = spectrum[2 * i + 1];
      mags.push(Math.sqrt(re * re + im * im));
    }

    // Apply mel filter bank
    const melEnergies = this.melFilters.map(filter =>
      filter.reduce((sum, w, k) => sum + w * mags[k], 0)
    );

    // Log scaling
    return melEnergies.map(e => Math.log(e + 1e-6));
  }

  /**
   * Create mel filter bank.
   * @returns Array of mel filters (each filter is an array of weights for FFT bins)
   */
  private createMelFilterBank(): number[][] {
    const filters: number[][] = [];
    const lowFreq = 0;
    const highFreq = this.sampleRate / 2;
    const melLow = this.hzToMel(lowFreq);
    const melHigh = this.hzToMel(highFreq);
    const melPoints = [];
    for (let i = 0; i <= this.melBands + 2; i++) {
      melPoints.push(melLow + (melHigh - melLow) * i / (this.melBands + 2));
    }
    const hzPoints = melPoints.map(mel => this.melToHz(mel));
    const binPoints = hzPoints.map(hz => Math.floor((this.fftSize + 1) * hz / this.sampleRate));

    for (let i = 0; i < this.melBands; i++) {
      const filter = new Array(this.fftSize / 2).fill(0);
      for (let j = binPoints[i]; j < binPoints[i + 1]; j++) {
        filter[j] = (j - binPoints[i]) / (binPoints[i + 1] - binPoints[i]);
      }
      for (let j = binPoints[i + 1]; j < binPoints[i + 2]; j++) {
        filter[j] = (binPoints[i + 2] - j) / (binPoints[i + 2] - binPoints[i + 1]);
      }
      filters.push(filter);
    }
    return filters;
  }

  /**
   * Convert Hz to mel scale.
   */
  private hzToMel(hz: number): number {
    return 2595 * Math.log10(1 + hz / 700);
  }

  /**
   * Convert mel scale to Hz.
   */
  private melToHz(mel: number): number {
    return 700 * (Math.pow(10, mel / 2595) - 1);
  }
}
