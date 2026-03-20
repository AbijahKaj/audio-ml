import FFT from 'fft.js';
import type { ComputeBackend } from '../compute/Backend';
import type { TensorHandle } from '../compute/types';
import type { FastConformerConfig } from '../model/ModelConfig';

export class FeaturePipeline {
  private backend: ComputeBackend;
  private fftSize: number;
  private windowSize: number;
  private hopSize: number;
  private numMelBands: number;
  private sampleRate: number;
  private fft: FFT;
  private melFilters: Float32Array[];
  private hannWindow: Float32Array;
  private preEmphasis: number = 0.97;

  constructor(config: FastConformerConfig, backend: ComputeBackend) {
    this.backend = backend;
    this.sampleRate = config.sampleRate;
    this.numMelBands = config.numMelBands;
    this.windowSize = Math.round(config.sampleRate * config.windowSizeMs / 1000);
    this.hopSize = Math.round(config.sampleRate * config.hopSizeMs / 1000);

    this.fftSize = 1;
    while (this.fftSize < this.windowSize) this.fftSize *= 2;

    this.fft = new FFT(this.fftSize);
    this.hannWindow = this.createHannWindow(this.windowSize);
    this.melFilters = this.createMelFilterBank();
  }

  extractFeatures(audio: Float32Array): TensorHandle {
    const emphasized = this.applyPreEmphasis(audio);
    const frames = this.frameSignal(emphasized);
    const numFrames = frames.length;
    const melFeatures = new Float32Array(numFrames * this.numMelBands);

    for (let f = 0; f < numFrames; f++) {
      const windowed = this.applyWindow(frames[f]);
      const paddedFrame = new Float32Array(this.fftSize);
      paddedFrame.set(windowed);

      const spectrum = this.fft.createComplexArray();
      this.fft.realTransform(spectrum, paddedFrame);
      this.fft.completeSpectrum(spectrum);

      const powerSpectrum = new Float32Array(this.fftSize / 2 + 1);
      for (let i = 0; i <= this.fftSize / 2; i++) {
        const re = spectrum[2 * i];
        const im = spectrum[2 * i + 1];
        powerSpectrum[i] = re * re + im * im;
      }

      for (let m = 0; m < this.numMelBands; m++) {
        let energy = 0;
        const filter = this.melFilters[m];
        for (let k = 0; k < filter.length; k++) {
          energy += filter[k] * powerSpectrum[k];
        }
        melFeatures[f * this.numMelBands + m] = Math.log(Math.max(energy, 1e-10));
      }
    }

    const features = this.backend.tensor(melFeatures, [1, numFrames, this.numMelBands]);
    return this.normalizeFeatures(features);
  }

  extractStreamingFeatures(audio: Float32Array): TensorHandle {
    return this.extractFeatures(audio);
  }

  get frameLength(): number {
    return this.windowSize;
  }

  get frameShift(): number {
    return this.hopSize;
  }

  get numBands(): number {
    return this.numMelBands;
  }

  private applyPreEmphasis(signal: Float32Array): Float32Array {
    const output = new Float32Array(signal.length);
    output[0] = signal[0];
    for (let i = 1; i < signal.length; i++) {
      output[i] = signal[i] - this.preEmphasis * signal[i - 1];
    }
    return output;
  }

  private frameSignal(signal: Float32Array): Float32Array[] {
    const frames: Float32Array[] = [];
    const numFrames = Math.max(0, Math.floor((signal.length - this.windowSize) / this.hopSize) + 1);
    for (let i = 0; i < numFrames; i++) {
      const start = i * this.hopSize;
      frames.push(signal.slice(start, start + this.windowSize));
    }
    return frames;
  }

  private applyWindow(frame: Float32Array): Float32Array {
    const windowed = new Float32Array(frame.length);
    for (let i = 0; i < frame.length; i++) {
      windowed[i] = frame[i] * this.hannWindow[i];
    }
    return windowed;
  }

  private createHannWindow(size: number): Float32Array {
    const window = new Float32Array(size);
    for (let i = 0; i < size; i++) {
      window[i] = 0.5 * (1 - Math.cos(2 * Math.PI * i / (size - 1)));
    }
    return window;
  }

  private normalizeFeatures(features: TensorHandle): TensorHandle {
    return this.backend.tidy(() => {
      const mean = this.backend.mean(features, [1], true);
      const centered = this.backend.sub(features, mean);
      const variance = this.backend.mean(
        this.backend.mul(centered, centered),
        [1],
        true
      );
      const std = this.backend.sqrt(this.backend.add(variance, this.backend.scalarTensor(1e-5)));
      return this.backend.div(centered, std);
    });
  }

  private createMelFilterBank(): Float32Array[] {
    const filters: Float32Array[] = [];
    const numBins = this.fftSize / 2 + 1;
    const lowFreq = 0;
    const highFreq = this.sampleRate / 2;
    const melLow = this.hzToMel(lowFreq);
    const melHigh = this.hzToMel(highFreq);

    const melPoints: number[] = [];
    for (let i = 0; i <= this.numMelBands + 1; i++) {
      melPoints.push(melLow + (melHigh - melLow) * i / (this.numMelBands + 1));
    }
    const hzPoints = melPoints.map(m => this.melToHz(m));
    const binPoints = hzPoints.map(hz => Math.floor((this.fftSize + 1) * hz / this.sampleRate));

    for (let i = 0; i < this.numMelBands; i++) {
      const filter = new Float32Array(numBins);
      const lower = binPoints[i];
      const center = binPoints[i + 1];
      const upper = binPoints[i + 2];

      for (let j = lower; j < center; j++) {
        if (center !== lower) {
          filter[j] = (j - lower) / (center - lower);
        }
      }
      for (let j = center; j < upper; j++) {
        if (upper !== center) {
          filter[j] = (upper - j) / (upper - center);
        }
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
