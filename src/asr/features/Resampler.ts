/**
 * Resampler — converts microphone input (44.1 kHz / 48 kHz) to 16 kHz
 * for the FastConformer feature pipeline.
 *
 * Uses linear interpolation by default. Adequate for speech (no high-frequency
 * content above 8 kHz matters for ASR). Upgrade to sinc if aliasing is audible.
 */
export class Resampler {
  private readonly fromRate: number;
  private readonly toRate: number;
  private readonly ratio: number;

  constructor(fromRate: number, toRate = 16000) {
    this.fromRate = fromRate;
    this.toRate = toRate;
    this.ratio = toRate / fromRate;
  }

  resample(input: Float32Array): Float32Array {
    if (this.fromRate === this.toRate) return input;

    const outputLen = Math.round(input.length * this.ratio);
    const output = new Float32Array(outputLen);
    const invRatio = 1 / this.ratio;

    for (let i = 0; i < outputLen; i++) {
      const srcIdx = i * invRatio;
      const lo = Math.floor(srcIdx);
      const hi = Math.min(lo + 1, input.length - 1);
      const frac = srcIdx - lo;
      output[i] = input[lo] * (1 - frac) + input[hi] * frac;
    }

    return output;
  }
}
