export class Resampler {
  constructor(
    private readonly fromRate: number,
    private readonly toRate: number = 16000,
  ) {}

  resample(input: Float32Array): Float32Array {
    if (this.fromRate === this.toRate) {
      return input;
    }

    const ratio = this.toRate / this.fromRate;
    const outputLength = Math.max(1, Math.round(input.length * ratio));
    const output = new Float32Array(outputLength);

    for (let index = 0; index < outputLength; index += 1) {
      const sourceIndex = index / ratio;
      const lo = Math.floor(sourceIndex);
      const hi = Math.min(lo + 1, input.length - 1);
      const frac = sourceIndex - lo;
      output[index] = input[lo] * (1 - frac) + input[hi] * frac;
    }

    return output;
  }
}
