import FFT from 'fft.js';
import type { ComputeBackend } from '../compute/Backend';
import type { FastConformerConfig } from '../model/ModelConfig';
import type { TensorHandle } from '../compute/types';

/**
 * NeMo-style log-mel features: pre-emphasis, Hann window, power spectrum, mel, log, optional CMVN.
 */
export class FeaturePipeline {
  private readonly fft: FFT;
  private readonly windowSize: number;
  private readonly hopSize: number;
  private readonly melBands: number;
  private readonly sampleRate: number;
  private readonly fftSize: number;
  private readonly melFilters: number[][];
  private readonly preemph = 0.97;
  private melMean?: Float32Array;
  private melStd?: Float32Array;

  constructor(
    config: FastConformerConfig,
    cmvn?: { mean: number[]; std: number[] },
  ) {
    this.sampleRate = config.sampleRate;
    this.windowSize = Math.round((config.sampleRate * config.windowSizeMs) / 1000);
    this.hopSize = Math.round((config.sampleRate * config.hopSizeMs) / 1000);
    this.melBands = config.numMelBands;
    this.fftSize = 2 ** Math.ceil(Math.log2(this.windowSize));
    this.fft = new FFT(this.fftSize);
    this.melFilters = this.createMelFilterBank();
    if (cmvn) {
      this.melMean = Float32Array.from(cmvn.mean);
      this.melStd = Float32Array.from(cmvn.std.map(s => Math.max(s, 1e-8)));
    }
  }

  extractFeatures(audio: Float32Array, backend: ComputeBackend): TensorHandle {
    const frames = this.frameAudio(audio);
    const n = frames.length;
    const f = this.melBands;
    const flat = new Float32Array(n * f);
    for (let i = 0; i < n; i++) {
      const m = this.melFrame(frames[i]!);
      const row = i * f;
      for (let j = 0; j < f; j++) {
        let v = m[j]!;
        if (this.melMean && this.melStd) {
          v = (v - this.melMean[j]!) / this.melStd[j]!;
        }
        flat[row + j] = v;
      }
    }
    return backend.tensor(flat, [1, n, f]);
  }

  private frameAudio(audio: Float32Array): Float32Array[] {
    const { windowSize, hopSize } = this;
    const frames: Float32Array[] = [];
    let pre = 0;
    for (let start = 0; start + windowSize <= audio.length; start += hopSize) {
      const w = new Float32Array(windowSize);
      for (let i = 0; i < windowSize; i++) {
        const x = audio[start + i]!;
        const p = i === 0 && start === 0 ? 0 : start + i > 0 ? audio[start + i - 1]! : pre;
        const y = x - this.preemph * (start + i === 0 ? 0 : p);
        w[i] = y * this.hann(i, windowSize);
      }
      pre = audio[start + windowSize - 1]!;
      frames.push(w);
    }
    return frames;
  }

  private hann(i: number, n: number): number {
    return 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (n - 1));
  }

  private melFrame(pcm: Float32Array): number[] {
    const padded = new Float32Array(this.fftSize);
    padded.set(pcm);
    const spectrum = this.fft.createComplexArray();
    this.fft.realTransform(spectrum, padded);
    this.fft.completeSpectrum(spectrum);
    const mags: number[] = [];
    for (let i = 0; i < this.fftSize / 2; i++) {
      const re = spectrum[2 * i]!;
      const im = spectrum[2 * i + 1]!;
      const mag = re * re + im * im;
      mags.push(mag);
    }
    const melEnergies = this.melFilters.map(filter =>
      filter.reduce((sum, w, k) => sum + w * (mags[k] ?? 0), 0),
    );
    const floor = 1e-10;
    return melEnergies.map(e => Math.log(Math.max(e, floor)));
  }

  private createMelFilterBank(): number[][] {
    const filters: number[][] = [];
    const lowFreq = 0;
    const highFreq = this.sampleRate / 2;
    const melLow = this.hzToMel(lowFreq);
    const melHigh = this.hzToMel(highFreq);
    const melPoints: number[] = [];
    for (let i = 0; i <= this.melBands + 2; i++) {
      melPoints.push(melLow + ((melHigh - melLow) * i) / (this.melBands + 2));
    }
    const hzPoints = melPoints.map(m => this.melToHz(m));
    const binPoints = hzPoints.map(hz => Math.floor(((this.fftSize + 1) * hz) / this.sampleRate));

    for (let i = 0; i < this.melBands; i++) {
      const filter = new Array(this.fftSize / 2).fill(0);
      for (let j = binPoints[i]!; j < binPoints[i + 1]!; j++) {
        const denom = binPoints[i + 1]! - binPoints[i]!;
        filter[j] = denom ? (j - binPoints[i]!) / denom : 0;
      }
      for (let j = binPoints[i + 1]!; j < binPoints[i + 2]!; j++) {
        const denom = binPoints[i + 2]! - binPoints[i + 1]!;
        filter[j] = denom ? (binPoints[i + 2]! - j) / denom : 0;
      }
      filters.push(filter);
    }
    return filters;
  }

  private hzToMel(hz: number): number {
    return 2595 * Math.log10(1 + hz / 700);
  }

  private melToHz(mel: number): number {
    return 700 * (Math.pow(10, mel / 2595) - 1);
  }
}
