import FFT from 'fft.js';
import type { ComputeBackend } from '../compute/Backend';
import type { TensorHandle } from '../compute/types';
import type { FastConformerConfig } from '../model/ModelConfig';

/**
 * NeMo-style log-mel features: pre-emphasis, Hann window, power mel, log, optional norm.
 */
export class FeaturePipeline {
  private readonly fftSize: number;
  private readonly hopSize: number;
  private readonly windowSize: number;
  private readonly fft: FFT;
  private readonly melFilters: number[][];
  private readonly preemph = 0.97;

  constructor(private readonly config: FastConformerConfig) {
    this.windowSize = Math.round((config.sampleRate * config.windowSizeMs) / 1000);
    this.hopSize = Math.round((config.sampleRate * config.hopSizeMs) / 1000);
    this.fftSize = 512;
    this.fft = new FFT(this.fftSize);
    this.melFilters = this.createMelFilterBank(config.numMelBands, config.sampleRate);
  }

  /**
   * @param audio mono PCM at `config.sampleRate`, any length ≥ windowSize
   * @returns tensor [1, T, numMelBands]
   */
  extractFeatures(backend: ComputeBackend, audio: Float32Array): TensorHandle {
    const { frames, n } = this.frameAudio(audio);
    const out = new Float32Array(n * this.config.numMelBands);
    const win = this.hann(this.windowSize);

    for (let f = 0; f < n; f++) {
      const frame = frames.subarray(f * this.windowSize, (f + 1) * this.windowSize);
      const windowed = new Float32Array(this.fftSize);
      for (let i = 0; i < this.windowSize; i++) {
        windowed[i] = frame[i]! * win[i]!;
      }

      const spectrum = this.fft.createComplexArray();
      this.fft.realTransform(spectrum, windowed);
      this.fft.completeSpectrum(spectrum);

      const mags: number[] = [];
      for (let i = 0; i < this.fftSize / 2; i++) {
        const re = spectrum[2 * i]!;
        const im = spectrum[2 * i + 1]!;
        const mag = re * re + im * im;
        mags.push(mag);
      }

      const mel = this.melFilters.map(filter =>
        filter.reduce((sum, w, k) => sum + w * (mags[k] ?? 0), 0),
      );
      const logMel = mel.map(e => Math.log(Math.max(e, 1e-10)));

      let norm = logMel;
      if (this.config.featureMean && this.config.featureStd) {
        norm = logMel.map(
          (v, i) => (v - (this.config.featureMean![i] ?? 0)) / (this.config.featureStd![i] ?? 1),
        );
      }

      for (let m = 0; m < this.config.numMelBands; m++) {
        out[f * this.config.numMelBands + m] = norm[m]!;
      }
    }

    return backend.tensor(out, [1, n, this.config.numMelBands]);
  }

  private frameAudio(audio: Float32Array): { frames: Float32Array; n: number } {
    const pre: Float32Array = new Float32Array(audio.length);
    pre[0] = audio[0] ?? 0;
    for (let i = 1; i < audio.length; i++) {
      pre[i] = audio[i]! - this.preemph * audio[i - 1]!;
    }
    const n = Math.max(0, Math.floor((pre.length - this.windowSize) / this.hopSize) + 1);
    const frames = new Float32Array(n * this.windowSize);
    for (let f = 0; f < n; f++) {
      const start = f * this.hopSize;
      for (let i = 0; i < this.windowSize; i++) {
        frames[f * this.windowSize + i] = pre[start + i] ?? 0;
      }
    }
    return { frames, n };
  }

  private hann(n: number): Float32Array {
    const w = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      w[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (n - 1)));
    }
    return w;
  }

  private createMelFilterBank(melBands: number, sampleRate: number): number[][] {
    const filters: number[][] = [];
    const lowFreq = 0;
    const highFreq = sampleRate / 2;
    const melLow = this.hzToMel(lowFreq);
    const melHigh = this.hzToMel(highFreq);
    const melPoints: number[] = [];
    for (let i = 0; i <= melBands + 2; i++) {
      melPoints.push(melLow + ((melHigh - melLow) * i) / (melBands + 2));
    }
    const hzPoints = melPoints.map(m => this.melToHz(m));
    const binPoints = hzPoints.map(hz => Math.floor(((this.fftSize + 1) * hz) / sampleRate));

    for (let i = 0; i < melBands; i++) {
      const filter = new Array(this.fftSize / 2).fill(0);
      for (let j = binPoints[i]!; j < binPoints[i + 1]!; j++) {
        filter[j] = (j - binPoints[i]!) / (binPoints[i + 1]! - binPoints[i]!);
      }
      for (let j = binPoints[i + 1]!; j < binPoints[i + 2]!; j++) {
        filter[j] = (binPoints[i + 2]! - j) / (binPoints[i + 2]! - binPoints[i + 1]!);
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