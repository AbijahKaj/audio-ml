import FFT from 'fft.js';
export interface ChromaFeaturesConfig { sampleRate: number, fftSize?: number }
export class ChromaFeaturesAnalyzer {
  private fft: FFT;
  private fftSize: number;
  private sampleRate: number;
  constructor(config: ChromaFeaturesConfig) {
    this.fftSize = config.fftSize || 1024;
    this.fft = new FFT(this.fftSize);
    this.sampleRate = config.sampleRate;
  }
  analyzeFrame(pcm: Float32Array): number[] {
    // 12 chroma bins (C, C#, D, ..., B)
    const chroma = new Array(12).fill(0);
    const spectrum = this.fft.createComplexArray();
    this.fft.realTransform(spectrum, pcm);
    this.fft.completeSpectrum(spectrum);
    for (let i = 0; i < this.fftSize / 2; i++) {
      const re = spectrum[2 * i];
      const im = spectrum[2 * i + 1];
      const mag = Math.sqrt(re * re + im * im);
      const freq = i * this.sampleRate / this.fftSize;
      const midi = 69 + 12 * Math.log2(freq / 440);
      const bin = Math.round(midi) % 12;
      if (bin >= 0 && bin < 12 && isFinite(bin)) {
        chroma[bin] += mag;
      }
    }
    // Normalize
    const sum = chroma.reduce((a, b) => a + b, 0) + 1e-12;
    return chroma.map(v => v / sum);
  }
}
