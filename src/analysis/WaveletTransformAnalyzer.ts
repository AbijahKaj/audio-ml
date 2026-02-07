export interface WaveletTransformConfig { sampleRate: number, levels?: number }
export class WaveletTransformAnalyzer {
  private levels: number;
  constructor(config: WaveletTransformConfig) {
    this.levels = config.levels || 1;
  }
  // Haar DWT (minimal, single-level or multi-level)
  analyzeFrame(pcm: Float32Array): Float32Array[] {
    let coeffs: Float32Array[] = [];
    let current = Float32Array.from(pcm);
    for (let l = 0; l < this.levels; l++) {
      const N = current.length;
      const approx = new Float32Array(N / 2);
      const detail = new Float32Array(N / 2);
      for (let i = 0; i < N; i += 2) {
        approx[i / 2] = (current[i] + current[i + 1]) / Math.sqrt(2);
        detail[i / 2] = (current[i] - current[i + 1]) / Math.sqrt(2);
      }
      coeffs.push(detail);
      current = approx;
    }
    coeffs.push(current); // final approx
    return coeffs;
  }
}
