import FFT from 'fft.js';
import type { ComputeBackend } from '../compute/Backend';
import type { TensorHandle } from '../compute/types';
import type { FastConformerConfig } from '../model/ModelConfig';

function nextPowerOfTwo(value: number): number {
  let power = 1;
  while (power < value) {
    power <<= 1;
  }
  return power;
}

function hzToMel(hz: number): number {
  return 2595 * Math.log10(1 + hz / 700);
}

function melToHz(mel: number): number {
  return 700 * (Math.pow(10, mel / 2595) - 1);
}

interface RunningStats {
  count: number;
  mean: Float32Array;
  m2: Float32Array;
}

/**
 * FastConformer feature extraction:
 * - pre-emphasis
 * - framed Hann-windowed STFT
 * - power mel filterbank
 * - log compression
 * - per-feature normalization
 */
export class FeaturePipeline {
  private readonly windowSize: number;
  private readonly hopSize: number;
  private readonly fftSize: number;
  private readonly melBands: number;
  private readonly hannWindow: Float32Array;
  private readonly melFilters: Float32Array[];
  private readonly fft: FFT;

  private streamBuffer = new Float32Array(0);
  private preEmphasisPrev = 0;
  private runningStats: RunningStats;

  constructor(
    config: FastConformerConfig,
    private backend: ComputeBackend,
  ) {
    this.windowSize = Math.round((config.sampleRate * config.windowSizeMs) / 1000);
    this.hopSize = Math.round((config.sampleRate * config.hopSizeMs) / 1000);
    this.fftSize = nextPowerOfTwo(this.windowSize);
    this.melBands = config.numMelBands;
    this.fft = new FFT(this.fftSize);
    this.hannWindow = this.createHannWindow(this.windowSize);
    this.melFilters = this.createMelFilterBank(this.fftSize, config.sampleRate, this.melBands);
    this.runningStats = {
      count: 0,
      mean: new Float32Array(this.melBands),
      m2: new Float32Array(this.melBands),
    };
  }

  extractFeatures(audio: Float32Array): TensorHandle {
    const preEmphasized = this.preEmphasize(audio, 0);
    const frames = this.frameSignal(preEmphasized);
    const features = this.framesToLogMel(frames);
    this.normalizeBatch(features);
    return this.toTensor(features);
  }

  extractStreamingFeatures(audioChunk: Float32Array): TensorHandle | null {
    if (audioChunk.length === 0) {
      return null;
    }

    const preEmphasized = this.preEmphasize(audioChunk, this.preEmphasisPrev);
    this.preEmphasisPrev = audioChunk[audioChunk.length - 1];

    const merged = new Float32Array(this.streamBuffer.length + preEmphasized.length);
    merged.set(this.streamBuffer, 0);
    merged.set(preEmphasized, this.streamBuffer.length);
    this.streamBuffer = merged;

    const frames: Float32Array[] = [];
    let offset = 0;
    while (offset + this.windowSize <= this.streamBuffer.length) {
      frames.push(this.streamBuffer.subarray(offset, offset + this.windowSize));
      offset += this.hopSize;
    }

    if (offset > 0) {
      this.streamBuffer = this.streamBuffer.slice(offset);
    }

    if (frames.length === 0) {
      return null;
    }

    const features = this.framesToLogMel(frames);
    this.normalizeStreaming(features);
    return this.toTensor(features);
  }

  reset(): void {
    this.streamBuffer = new Float32Array(0);
    this.preEmphasisPrev = 0;
    this.runningStats = {
      count: 0,
      mean: new Float32Array(this.melBands),
      m2: new Float32Array(this.melBands),
    };
  }

  private preEmphasize(input: Float32Array, previous: number): Float32Array {
    const output = new Float32Array(input.length);
    let prev = previous;
    for (let i = 0; i < input.length; i++) {
      const sample = input[i];
      output[i] = sample - 0.97 * prev;
      prev = sample;
    }
    return output;
  }

  private frameSignal(input: Float32Array): Float32Array[] {
    if (input.length < this.windowSize) {
      return [];
    }

    const frameCount = 1 + Math.floor((input.length - this.windowSize) / this.hopSize);
    const frames: Float32Array[] = [];
    for (let index = 0; index < frameCount; index++) {
      const start = index * this.hopSize;
      frames.push(input.subarray(start, start + this.windowSize));
    }
    return frames;
  }

  private framesToLogMel(frames: Float32Array[]): Float32Array[] {
    const output: Float32Array[] = [];
    const fftInput = new Float32Array(this.fftSize);
    const fftOutput = this.fft.createComplexArray();
    const bins = this.fftSize / 2 + 1;
    const powerSpectrum = new Float32Array(bins);

    for (const frame of frames) {
      fftInput.fill(0);
      for (let i = 0; i < this.windowSize; i++) {
        fftInput[i] = frame[i] * this.hannWindow[i];
      }

      this.fft.realTransform(fftOutput, fftInput);
      this.fft.completeSpectrum(fftOutput);

      for (let i = 0; i < bins; i++) {
        const re = fftOutput[2 * i];
        const im = fftOutput[2 * i + 1];
        powerSpectrum[i] = re * re + im * im;
      }

      const mel = new Float32Array(this.melBands);
      for (let m = 0; m < this.melBands; m++) {
        const filter = this.melFilters[m];
        let energy = 0;
        for (let k = 0; k < bins; k++) {
          energy += filter[k] * powerSpectrum[k];
        }
        mel[m] = Math.log(energy + 1e-6);
      }
      output.push(mel);
    }

    return output;
  }

  private normalizeBatch(frames: Float32Array[]): void {
    if (frames.length === 0) return;
    const means = new Float32Array(this.melBands);
    const vars = new Float32Array(this.melBands);

    for (const frame of frames) {
      for (let i = 0; i < this.melBands; i++) {
        means[i] += frame[i];
      }
    }
    for (let i = 0; i < this.melBands; i++) {
      means[i] /= frames.length;
    }

    for (const frame of frames) {
      for (let i = 0; i < this.melBands; i++) {
        const diff = frame[i] - means[i];
        vars[i] += diff * diff;
      }
    }
    for (let i = 0; i < this.melBands; i++) {
      vars[i] = Math.sqrt(vars[i] / frames.length + 1e-5);
    }

    for (const frame of frames) {
      for (let i = 0; i < this.melBands; i++) {
        frame[i] = (frame[i] - means[i]) / vars[i];
      }
    }
  }

  private normalizeStreaming(frames: Float32Array[]): void {
    for (const frame of frames) {
      this.runningStats.count += 1;
      const count = this.runningStats.count;
      for (let i = 0; i < this.melBands; i++) {
        const value = frame[i];
        const delta = value - this.runningStats.mean[i];
        this.runningStats.mean[i] += delta / count;
        const delta2 = value - this.runningStats.mean[i];
        this.runningStats.m2[i] += delta * delta2;
      }
    }

    const denom = Math.max(1, this.runningStats.count - 1);
    for (const frame of frames) {
      for (let i = 0; i < this.melBands; i++) {
        const variance = this.runningStats.m2[i] / denom;
        const std = Math.sqrt(variance + 1e-5);
        frame[i] = (frame[i] - this.runningStats.mean[i]) / std;
      }
    }
  }

  private toTensor(frames: Float32Array[]): TensorHandle {
    if (frames.length === 0) {
      return this.backend.zeros([1, 0, this.melBands]);
    }

    const flattened = new Float32Array(frames.length * this.melBands);
    for (let t = 0; t < frames.length; t++) {
      flattened.set(frames[t], t * this.melBands);
    }
    return this.backend.tensor(flattened, [1, frames.length, this.melBands]);
  }

  private createHannWindow(length: number): Float32Array {
    const window = new Float32Array(length);
    for (let i = 0; i < length; i++) {
      window[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (length - 1));
    }
    return window;
  }

  private createMelFilterBank(fftSize: number, sampleRate: number, melBands: number): Float32Array[] {
    const bins = fftSize / 2 + 1;
    const lowMel = hzToMel(0);
    const highMel = hzToMel(sampleRate / 2);
    const melPoints = new Float32Array(melBands + 2);
    for (let i = 0; i < melPoints.length; i++) {
      melPoints[i] = lowMel + ((highMel - lowMel) * i) / (melBands + 1);
    }

    const hzPoints = Array.from(melPoints, (mel) => melToHz(mel));
    const binPoints = hzPoints.map((hz) => Math.floor(((fftSize + 1) * hz) / sampleRate));

    const filters: Float32Array[] = [];
    for (let band = 0; band < melBands; band++) {
      const filter = new Float32Array(bins);
      const left = binPoints[band];
      const center = binPoints[band + 1];
      const right = binPoints[band + 2];

      for (let bin = left; bin < center && bin < bins; bin++) {
        filter[bin] = (bin - left) / Math.max(1, center - left);
      }
      for (let bin = center; bin < right && bin < bins; bin++) {
        filter[bin] = (right - bin) / Math.max(1, right - center);
      }

      filters.push(filter);
    }

    return filters;
  }
}
