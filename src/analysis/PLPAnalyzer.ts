/**
 * PLPAnalyzer extracts Perceptual Linear Prediction (PLP) coefficients from PCM audio frames.
 * PLP is used in speech processing to model the auditory spectrum using perceptual principles.
 * This implementation includes critical band analysis, equal loudness pre-emphasis, and LPC analysis.
 * Usage: const plp = new PLPAnalyzer({ sampleRate: 16000 }); plp.analyzeFrame(pcm)
 */
import FFT from 'fft.js';

export interface PLPConfig { 
  sampleRate: number;
  fftSize?: number;
  order?: number;
  numBands?: number;
}

export class PLPAnalyzer {
  private fft: FFT;
  private fftSize: number;
  private order: number;
  private numBands: number;
  private sampleRate: number;
  private barkFilters: number[][];

  /**
   * @param config - Configuration with sample rate, optional FFT size, LPC order, and number of critical bands
   */
  constructor(config: PLPConfig) {
    this.fftSize = config.fftSize || 512;
    this.order = config.order || 12;
    this.numBands = config.numBands || 18;
    this.sampleRate = config.sampleRate;
    this.fft = new FFT(this.fftSize);
    this.barkFilters = this.createBarkFilterBank();
  }

  /**
   * Extract PLP coefficients from a PCM frame.
   * @param pcm - Input PCM audio frame
   * @returns Array of PLP coefficients
   */
  analyzeFrame(pcm: Float32Array): number[] {
    // Ensure correct frame size
    if (pcm.length !== this.fftSize) {
      throw new Error(`PCM frame must be fftSize (${this.fftSize}) samples, got ${pcm.length}`);
    }
    // Step 1: Compute power spectrum
    const spectrum = this.fft.createComplexArray();
    this.fft.realTransform(spectrum, pcm);
    this.fft.completeSpectrum(spectrum);
    
    const powerSpectrum: number[] = [];
    for (let i = 0; i < this.fftSize / 2; i++) {
      const re = spectrum[2 * i];
      const im = spectrum[2 * i + 1];
      powerSpectrum.push(re * re + im * im);
    }

    // Step 2: Critical band analysis (Bark scale)
    const barkEnergies = this.barkFilters.map(filter => 
      filter.reduce((sum, w, k) => sum + w * powerSpectrum[k], 0)
    );

    // Step 3: Equal loudness pre-emphasis
    const preemphasized = barkEnergies.map((energy, i) => {
      const bark = i * 1.0; // Approximate bark value
      const equalLoudness = this.equalLoudnessWeight(bark);
      return energy * equalLoudness;
    });

    // Step 4: Intensity-loudness conversion (cube root)
    const loudness = preemphasized.map(e => Math.pow(e + 1e-12, 1.0 / 3.0));

    // Step 5: Inverse FFT to get autocorrelation
    const autocorr = this.barkToAutocorr(loudness);

    // Step 6: LPC analysis using Levinson-Durbin
    return this.levinsonDurbin(autocorr, this.order);
  }

  /**
   * Create a Bark scale filter bank.
   * @returns Array of Bark filters
   */
  private createBarkFilterBank(): number[][] {
    const filters: number[][] = [];
    const nyquist = this.sampleRate / 2;
    const maxBark = this.hzToBark(nyquist);
    
    for (let band = 0; band < this.numBands; band++) {
      const filter = new Array(this.fftSize / 2).fill(0);
      const centerBark = (band + 0.5) * maxBark / this.numBands;
      
      for (let i = 0; i < this.fftSize / 2; i++) {
        const freq = (i * this.sampleRate) / this.fftSize;
        const bark = this.hzToBark(freq);
        const distance = Math.abs(bark - centerBark);
        
        // Triangular filter shape
        if (distance < 1.0) {
          filter[i] = 1.0 - distance;
        }
      }
      
      // Normalize
      const sum = filter.reduce((a, b) => a + b, 0);
      if (sum > 0) {
        for (let i = 0; i < filter.length; i++) {
          filter[i] /= sum;
        }
      }
      
      filters.push(filter);
    }
    
    return filters;
  }

  /**
   * Convert Hz to Bark scale.
   */
  private hzToBark(hz: number): number {
    return 13 * Math.atan(0.00076 * hz) + 3.5 * Math.atan(Math.pow(hz / 7500, 2));
  }

  /**
   * Equal loudness pre-emphasis weight.
   */
  private equalLoudnessWeight(bark: number): number {
    // Simplified equal loudness curve
    const w = 1.0 / (1.0 + Math.pow(bark / 15.0, 2));
    return w;
  }

  /**
   * Convert Bark domain loudness to autocorrelation via inverse FFT.
   * Interpolates Bark domain loudness back to frequency domain, then computes autocorrelation.
   */
  private barkToAutocorr(loudness: number[]): number[] {
    // Interpolate Bark domain loudness back to frequency domain
    const nyquist = this.sampleRate / 2;
    const maxBark = this.hzToBark(nyquist);
    const freqSpectrum = new Array(this.fftSize / 2).fill(0);
    
    for (let i = 0; i < this.fftSize / 2; i++) {
      const freq = (i * this.sampleRate) / this.fftSize;
      const bark = this.hzToBark(freq);
      const bandIndex = (bark / maxBark) * this.numBands;
      
      // Linear interpolation
      const lowerBand = Math.floor(bandIndex);
      const upperBand = Math.min(Math.ceil(bandIndex), this.numBands - 1);
      const t = bandIndex - lowerBand;
      
      if (lowerBand >= 0 && lowerBand < loudness.length && upperBand >= 0 && upperBand < loudness.length) {
        freqSpectrum[i] = loudness[lowerBand] * (1 - t) + loudness[upperBand] * t;
      } else if (lowerBand >= 0 && lowerBand < loudness.length) {
        freqSpectrum[i] = loudness[lowerBand];
      }
    }
    
    // Pad to fftSize for IFFT (mirror for real signal)
    const padded = new Array(this.fftSize).fill(0);
    for (let i = 0; i < this.fftSize / 2; i++) {
      padded[i] = freqSpectrum[i];
      if (i > 0) {
        padded[this.fftSize - i] = freqSpectrum[i]; // Mirror for real IFFT
      }
    }
    
    // Inverse FFT to get autocorrelation
    const input = this.fft.createComplexArray();
    const output = this.fft.createComplexArray();
    for (let i = 0; i < padded.length; i++) {
      input[2 * i] = padded[i];
      input[2 * i + 1] = 0;
    }
    this.fft.inverseTransform(output, input);
    
    // Extract autocorrelation (first order+1 values)
    const autocorr = new Array(this.order + 1).fill(0);
    for (let i = 0; i <= this.order; i++) {
      autocorr[i] = output[2 * i] / this.fftSize;
    }
    
    return autocorr;
  }

  /**
   * Levinson-Durbin recursion for LPC coefficients.
   */
  private levinsonDurbin(R: number[], order: number): number[] {
    const a = new Array(order + 1).fill(0);
    const e = new Array(order + 1).fill(0);
    a[0] = 1;
    e[0] = R[0];
    
    for (let i = 1; i <= order; i++) {
      let acc = 0;
      for (let j = 1; j < i; j++) {
        acc += a[j] * R[i - j];
      }
      const k = (R[i] - acc) / (e[i - 1] + 1e-12);
      a[i] = k;
      for (let j = 1; j < i; j++) {
        a[j] = a[j] - k * a[i - j];
      }
      e[i] = (1 - k * k) * e[i - 1];
    }
    
    return a.slice(1);
  }
}
