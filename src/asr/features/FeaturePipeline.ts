import { MelSpectrogramAnalyzer } from '../../analysis/MelSpectrogramAnalyzer';
import type { ComputeBackend } from '../compute/Backend';
import type { TensorHandle } from '../compute/types';
import type { FastConformerConfig } from '../model/ModelConfig';

function nextPowerOfTwo(value: number): number {
  return 2 ** Math.ceil(Math.log2(value));
}

function concatFloat32(chunks: Float32Array[]): Float32Array {
  const totalLength = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const output = new Float32Array(totalLength);
  let offset = 0;
  for (const chunk of chunks) {
    output.set(chunk, offset);
    offset += chunk.length;
  }
  return output;
}

export class FeaturePipeline {
  private readonly analyzer: MelSpectrogramAnalyzer;
  private readonly windowSize: number;
  private readonly hopSize: number;
  private readonly fftSize: number;
  private readonly numMelBands: number;
  private readonly hannWindow: Float32Array;
  private residual = new Float32Array(0);
  private previousSample = 0;

  constructor(
    config: FastConformerConfig,
    private readonly backend: ComputeBackend,
  ) {
    this.windowSize = Math.round(config.sampleRate * config.windowSizeMs / 1000);
    this.hopSize = Math.round(config.sampleRate * config.hopSizeMs / 1000);
    this.fftSize = nextPowerOfTwo(this.windowSize);
    this.numMelBands = config.numMelBands;
    this.hannWindow = new Float32Array(this.windowSize);

    for (let index = 0; index < this.windowSize; index += 1) {
      this.hannWindow[index] = 0.5 - 0.5 * Math.cos((2 * Math.PI * index) / Math.max(1, this.windowSize - 1));
    }

    this.analyzer = new MelSpectrogramAnalyzer({
      sampleRate: config.sampleRate,
      fftSize: this.fftSize,
      melBands: this.numMelBands,
    });
  }

  reset(): void {
    this.residual = new Float32Array(0);
    this.previousSample = 0;
  }

  extractFeatures(audio: Float32Array): TensorHandle {
    const frames = this.extractFeatureFrames(audio);
    return this.framesToTensor(frames);
  }

  consume(audio: Float32Array): number[][] {
    const combined = concatFloat32([this.residual, audio]);
    const frames: number[][] = [];
    let frameStart = 0;

    while (frameStart + this.windowSize <= combined.length) {
      const window = combined.subarray(frameStart, frameStart + this.windowSize);
      frames.push(this.frameToMel(window));
      frameStart += this.hopSize;
    }

    this.residual = combined.slice(frameStart);
    return frames;
  }

  framesToTensor(frames: number[][]): TensorHandle {
    const frameCount = Math.max(1, frames.length);
    const flat = new Float32Array(frameCount * this.numMelBands);

    if (frames.length === 0) {
      return this.backend.tensor(flat, [1, frameCount, this.numMelBands]);
    }

    for (let frameIndex = 0; frameIndex < frames.length; frameIndex += 1) {
      flat.set(frames[frameIndex], frameIndex * this.numMelBands);
    }

    return this.backend.tensor(flat, [1, frameCount, this.numMelBands]);
  }

  extractFeatureFrames(audio: Float32Array): number[][] {
    this.reset();
    const frames = this.consume(audio);
    this.reset();
    return frames;
  }

  private frameToMel(frame: Float32Array): number[] {
    const emphasized = new Float32Array(this.windowSize);
    for (let index = 0; index < frame.length; index += 1) {
      const current = frame[index];
      emphasized[index] = (current - 0.97 * this.previousSample) * this.hannWindow[index];
      this.previousSample = current;
    }

    const padded = new Float32Array(this.fftSize);
    padded.set(emphasized);
    const mel = this.analyzer.analyzeFrame(padded);
    const mean = mel.reduce((sum, value) => sum + value, 0) / mel.length;
    const variance = mel.reduce((sum, value) => sum + ((value - mean) ** 2), 0) / mel.length;
    const std = Math.sqrt(Math.max(variance, 1e-6));
    return mel.map((value) => (value - mean) / std);
  }
}
