// Minimal autocorrelation method for LPC coefficients
export interface LPCConfig { sampleRate: number, order?: number }
export class LPCAnalyzer {
  private order: number;
  constructor(config: LPCConfig) {
    this.order = config.order || 12;
  }
  analyzeFrame(pcm: Float32Array): number[] {
    // Autocorrelation method
    const R = new Array(this.order + 1).fill(0);
    for (let lag = 0; lag <= this.order; lag++) {
      for (let i = 0; i < pcm.length - lag; i++) {
        R[lag] += pcm[i] * pcm[i + lag];
      }
    }
    // Levinson-Durbin recursion
    const a = new Array(this.order + 1).fill(0);
    const e = new Array(this.order + 1).fill(0);
    a[0] = 1;
    e[0] = R[0];
    for (let i = 1; i <= this.order; i++) {
      let acc = 0;
      for (let j = 1; j < i; j++) {
        acc += a[j] * R[i - j];
      }
      const k = (R[i] - acc) / (e[i - 1] + 1e-12);
      a[i] = k;
      for (let j = 1; j < i; j++) {
        a[j] = a[j] - k * a[i - j];
      }
      e[i] = (1 - k * k) * e[i - 1];
    }
    return a.slice(1);
  }
}
