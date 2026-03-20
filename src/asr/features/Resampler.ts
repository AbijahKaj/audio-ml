/**
 * Lightweight linear-interpolation resampler for speech use-cases.
 */
export class Resampler {
  private readonly ratio: number;
  private carrySourcePosition = 0;
  private carryLastSample = 0;
  private hasCarrySample = false;

  constructor(
    private fromRate: number,
    private toRate: number = 16000,
  ) {
    if (fromRate <= 0 || toRate <= 0) {
      throw new Error(`Invalid sample rates: fromRate=${fromRate}, toRate=${toRate}`);
    }
    this.ratio = fromRate / toRate;
  }

  /**
   * Stateless conversion. Best for offline/full-buffer audio.
   */
  resample(input: Float32Array): Float32Array {
    if (this.fromRate === this.toRate || input.length === 0) {
      return input;
    }

    const outputLength = Math.max(1, Math.round(input.length * (this.toRate / this.fromRate)));
    const output = new Float32Array(outputLength);

    for (let i = 0; i < outputLength; i++) {
      const sourceIndex = i * this.ratio;
      const lo = Math.floor(sourceIndex);
      const hi = Math.min(lo + 1, input.length - 1);
      const frac = sourceIndex - lo;
      output[i] = input[lo] * (1 - frac) + input[hi] * frac;
    }

    return output;
  }

  /**
   * Stateful conversion for chunked streaming input.
   */
  resampleStreaming(input: Float32Array): Float32Array {
    if (this.fromRate === this.toRate || input.length === 0) {
      return input;
    }

    const firstSourceIndex = this.carrySourcePosition;
    const approxLength = Math.max(0, Math.floor((input.length - firstSourceIndex) / this.ratio));
    const output: number[] = [];
    let sourceIndex = firstSourceIndex;

    if (approxLength > 0) {
      output.length = 0;
    }

    while (sourceIndex < input.length) {
      const lo = Math.floor(sourceIndex);
      const hi = lo + 1;
      const frac = sourceIndex - lo;

      const a = this.sampleAt(input, lo);
      const b = this.sampleAt(input, hi);
      output.push(a * (1 - frac) + b * frac);
      sourceIndex += this.ratio;
    }

    this.carrySourcePosition = sourceIndex - input.length;
    this.carryLastSample = input[input.length - 1];
    this.hasCarrySample = true;
    return Float32Array.from(output);
  }

  reset(): void {
    this.carrySourcePosition = 0;
    this.carryLastSample = 0;
    this.hasCarrySample = false;
  }

  private sampleAt(input: Float32Array, index: number): number {
    if (index < 0) {
      return this.hasCarrySample ? this.carryLastSample : input[0];
    }

    if (index >= input.length) {
      return input[input.length - 1];
    }

    return input[index];
  }
}
