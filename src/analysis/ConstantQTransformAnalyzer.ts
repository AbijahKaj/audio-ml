/**
 * ConstantQTransformAnalyzer computes the Constant-Q Transform (CQT) of a PCM audio frame.
 * CQT is a time-frequency analysis method with logarithmically spaced frequency bins, useful for music analysis.
 * The Q factor (quality factor) remains constant across all frequency bins.
 * Usage: const cqt = new ConstantQTransformAnalyzer({ sampleRate: 16000 }); cqt.analyzeFrame(pcm)
 */
import FFT from 'fft.js';

export interface ConstantQTransformConfig { 
  sampleRate: number;
  fftSize?: number;
  binsPerOctave?: number;
  minFreq?: number;
  maxFreq?: number;
}

export class ConstantQTransformAnalyzer {
  private fft: FFT;
  private fftSize: number;
  private sampleRate: number;
  private binsPerOctave: number;
  private minFreq: number;
  private maxFreq: number;
  private kernels: Array<{ centerFreq: number; kernel: Float32Array }>;

  /**
   * @param config - Configuration with sample rate, optional FFT size, bins per octave, and frequency range
   */
  constructor(config: ConstantQTransformConfig) {
    this.fftSize = config.fftSize || 2048;
    this.sampleRate = config.sampleRate;
    this.binsPerOctave = config.binsPerOctave || 12;
    this.minFreq = config.minFreq || 27.5; // A0
    this.maxFreq = config.maxFreq || this.sampleRate / 2;
    this.fft = new FFT(this.fftSize);
    this.kernels = this.createCQTKernels();
  }

  /**
   * Compute the Constant-Q Transform for a PCM frame.
   * @param pcm - Input PCM audio frame
   * @returns CQT coefficients as Float32Array
   */
  analyzeFrame(pcm: Float32Array): Float32Array {
    // Pad or truncate input to fftSize
    const padded = new Float32Array(this.fftSize);
    const copyLen = Math.min(pcm.length, this.fftSize);
    padded.set(pcm.subarray(0, copyLen), 0);

    // Apply window (Hann window)
    const windowed = new Float32Array(this.fftSize);
    for (let i = 0; i < this.fftSize; i++) {
      const window = 0.5 * (1 - Math.cos(2 * Math.PI * i / (this.fftSize - 1)));
      windowed[i] = padded[i] * window;
    }

    // Compute FFT
    const spectrum = this.fft.createComplexArray();
    this.fft.realTransform(spectrum, windowed);
    this.fft.completeSpectrum(spectrum);

    // Compute CQT by convolving with kernels
    const cqt = new Float32Array(this.kernels.length);
    for (let k = 0; k < this.kernels.length; k++) {
      const kernel = this.kernels[k].kernel;
      let realSum = 0;
      let imagSum = 0;
      
      for (let i = 0; i < kernel.length && i < this.fftSize / 2; i++) {
        const re = spectrum[2 * i];
        const im = spectrum[2 * i + 1];
        realSum += re * kernel[i];
        imagSum += im * kernel[i];
      }
      
      cqt[k] = Math.sqrt(realSum * realSum + imagSum * imagSum);
    }

    return cqt;
  }

  /**
   * Create CQT kernels for each frequency bin.
   */
  private createCQTKernels(): Array<{ centerFreq: number; kernel: Float32Array }> {
    const kernels: Array<{ centerFreq: number; kernel: Float32Array }> = [];
    const Q = 1.0 / (Math.pow(2, 1.0 / this.binsPerOctave) - 1.0);
    
    // Calculate number of bins
    const numBins = Math.floor(
      this.binsPerOctave * Math.log2(this.maxFreq / this.minFreq)
    );
    
    for (let bin = 0; bin < numBins; bin++) {
      const centerFreq = this.minFreq * Math.pow(2, bin / this.binsPerOctave);
      
      if (centerFreq > this.maxFreq) break;
      
      // Bandwidth is proportional to center frequency (constant Q)
      const bandwidth = centerFreq / Q;
      
      // Create kernel in frequency domain
      const kernel = new Float32Array(this.fftSize / 2);
      
      // Gaussian-like window in frequency domain
      for (let i = 0; i < kernel.length; i++) {
        const freq = (i * this.sampleRate) / this.fftSize;
        const distance = Math.abs(freq - centerFreq);
        if (distance < bandwidth) {
          const sigma = bandwidth / 2.0;
          kernel[i] = Math.exp(-0.5 * Math.pow(distance / sigma, 2));
        }
      }
      
      // Normalize
      const sum = kernel.reduce((a, b) => a + b, 0);
      if (sum > 0) {
        for (let i = 0; i < kernel.length; i++) {
          kernel[i] /= sum;
        }
      }
      
      kernels.push({ centerFreq, kernel });
    }
    
    return kernels;
  }
}
