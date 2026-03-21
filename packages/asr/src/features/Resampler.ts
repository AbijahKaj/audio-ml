export class Resampler {
  private fromRate: number;
  private toRate: number;

  constructor(fromRate: number, toRate: number = 16000) {
    this.fromRate = fromRate;
    this.toRate = toRate;
  }

  resample(input: Float32Array): Float32Array {
    if (this.fromRate === this.toRate) return input;

    const ratio = this.toRate / this.fromRate;
    const outputLen = Math.round(input.length * ratio);
    const output = new Float32Array(outputLen);

    for (let i = 0; i < outputLen; i++) {
      const srcIdx = i / ratio;
      const lo = Math.floor(srcIdx);
      const hi = Math.min(lo + 1, input.length - 1);
      const frac = srcIdx - lo;
      output[i] = input[lo] * (1 - frac) + input[hi] * frac;
    }

    return output;
  }
}
