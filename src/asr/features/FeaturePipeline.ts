import FFT from 'fft.js';
import type { FastConformerConfig } from '../model/ModelConfig.js';

/**
 * Log-mel feature pipeline matching NeMo's AudioToMelSpectrogramPreprocessor exactly.
 *
 * Key differences from the existing MelSpectrogramAnalyzer:
 *  - Power spectrum (mag² not mag) before mel filtering
 *  - Hann window applied per frame before FFT
 *  - Handles overlapping windowing / hopping internally
 *  - Clamp before log (max(1e-10, ...)) matching NeMo's log_zero_guard_value
 *  - Optional per-utterance mean/var normalization (CMVN)
 *
 * Output: Float32Array of shape [numFrames * numMelBands]
 * Interpreted as [numFrames, numMelBands] — the caller reshapes for the encoder.
 */
export class FeaturePipeline {
  private readonly windowSize: number;  // samples
  private readonly hopSize: number;     // samples
  private readonly fftSize: number;     // next power-of-2 >= windowSize
  private readonly numMelBands: number;
  private readonly melFilters: Float32Array[];
  private readonly hannWindow: Float32Array;
  private readonly fft: FFT;

  constructor(config: FastConformerConfig) {
    this.windowSize = Math.round(config.sampleRate * config.windowSizeMs / 1000);
    this.hopSize = Math.round(config.sampleRate * config.hopSizeMs / 1000);
    this.numMelBands = config.numMelBands;

    // FFT size = smallest power-of-2 >= windowSize
    this.fftSize = nextPow2(this.windowSize);
    this.fft = new FFT(this.fftSize);

    this.hannWindow = buildHannWindow(this.windowSize);
    this.melFilters = buildMelFilterBank(
      this.fftSize,
      config.sampleRate,
      config.numMelBands,
      0,                           // lowFreqHz (NeMo default = 0)
      config.sampleRate / 2,       // highFreqHz = Nyquist
    );
  }

  /**
   * Convert raw PCM audio → log-mel features.
   *
   * @param audio  16 kHz mono float32 samples
   * @returns      Flat Float32Array of length [numFrames × numMelBands];
   *               interpret as row-major [numFrames][numMelBands]
   */
  extractFeatures(audio: Float32Array): { features: Float32Array; numFrames: number } {
    const numFrames = Math.floor((audio.length - this.windowSize) / this.hopSize) + 1;
    const numBands = this.numMelBands;
    const features = new Float32Array(numFrames * numBands);

    const paddedFrame = new Float32Array(this.fftSize);
    const complexOut = this.fft.createComplexArray();

    for (let t = 0; t < numFrames; t++) {
      const start = t * this.hopSize;

      // Fill padded frame (zero-pad if shorter than fftSize)
      paddedFrame.fill(0);
      for (let k = 0; k < this.windowSize; k++) {
        const s = start + k;
        paddedFrame[k] = s < audio.length
          ? audio[s] * this.hannWindow[k]
          : 0;
      }

      // FFT
      this.fft.realTransform(complexOut, paddedFrame);
      this.fft.completeSpectrum(complexOut);

      // Power spectrum: |X[k]|² for k = 0..fftSize/2
      const numBins = this.fftSize / 2 + 1;
      const power = new Float32Array(numBins);
      for (let k = 0; k < numBins; k++) {
        const re = complexOut[2 * k];
        const im = complexOut[2 * k + 1];
        power[k] = re * re + im * im;
      }

      // Apply mel filter bank and take log
      const frameOffset = t * numBands;
      for (let m = 0; m < numBands; m++) {
        const filter = this.melFilters[m];
        let energy = 0;
        for (let k = 0; k < filter.length; k++) {
          energy += filter[k] * power[k];
        }
        features[frameOffset + m] = Math.log(Math.max(energy, 1e-10));
      }
    }

    // Per-utterance mean normalization (matches NeMo's normalize='per_feature')
    applyPerFeatureNorm(features, numFrames, numBands);

    return { features, numFrames };
  }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function nextPow2(n: number): number {
  let p = 1;
  while (p < n) p <<= 1;
  return p;
}

function buildHannWindow(size: number): Float32Array {
  const w = new Float32Array(size);
  for (let i = 0; i < size; i++) {
    w[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (size - 1)));
  }
  return w;
}

/**
 * Build a triangular mel filter bank matching NeMo's implementation.
 * Returns an array of numBands Float32Arrays, each of length fftSize/2+1.
 */
function buildMelFilterBank(
  fftSize: number,
  sampleRate: number,
  numBands: number,
  lowFreqHz: number,
  highFreqHz: number,
): Float32Array[] {
  const numBins = fftSize / 2 + 1;
  const melLow = hzToMel(lowFreqHz);
  const melHigh = hzToMel(highFreqHz);

  // numBands + 2 equally-spaced mel points → converted back to Hz → FFT bin
  const melPoints: number[] = [];
  for (let i = 0; i <= numBands + 1; i++) {
    melPoints.push(melLow + (melHigh - melLow) * i / (numBands + 1));
  }
  const hzPoints = melPoints.map(melToHz);
  const bins = hzPoints.map(hz => Math.floor((numBins - 1) * hz / (sampleRate / 2)));

  const filters: Float32Array[] = [];
  for (let m = 1; m <= numBands; m++) {
    const filter = new Float32Array(numBins);
    for (let k = bins[m - 1]; k < bins[m]; k++) {
      filter[k] = (k - bins[m - 1]) / (bins[m] - bins[m - 1]);
    }
    for (let k = bins[m]; k <= bins[m + 1]; k++) {
      filter[k] = (bins[m + 1] - k) / (bins[m + 1] - bins[m]);
    }
    filters.push(filter);
  }

  return filters;
}

function hzToMel(hz: number): number {
  return 2595 * Math.log10(1 + hz / 700);
}

function melToHz(mel: number): number {
  return 700 * (Math.pow(10, mel / 2595) - 1);
}

/**
 * Per-feature (per mel band) mean subtraction across time.
 * Modifies `features` in-place.
 */
function applyPerFeatureNorm(
  features: Float32Array,
  numFrames: number,
  numBands: number,
): void {
  for (let m = 0; m < numBands; m++) {
    let sum = 0;
    for (let t = 0; t < numFrames; t++) {
      sum += features[t * numBands + m];
    }
    const mean = sum / numFrames;

    let varSum = 0;
    for (let t = 0; t < numFrames; t++) {
      const diff = features[t * numBands + m] - mean;
      varSum += diff * diff;
    }
    const std = Math.sqrt(varSum / numFrames + 1e-5);

    for (let t = 0; t < numFrames; t++) {
      features[t * numBands + m] = (features[t * numBands + m] - mean) / std;
    }
  }
}
