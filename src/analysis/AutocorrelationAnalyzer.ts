/**
 * AutocorrelationAnalyzer computes the autocorrelation function of a PCM audio frame.
 * Autocorrelation is useful for pitch detection and periodicity analysis in audio signals.
 * Usage: const ac = new AutocorrelationAnalyzer({ sampleRate: 16000 }); ac.analyzeFrame(pcm)
 */
export interface AutocorrelationConfig { sampleRate: number, maxLag?: number }

export class AutocorrelationAnalyzer {
  private maxLag: number;
  /**
   * @param config - Configuration with sample rate and optional max lag
   */
  constructor(config: AutocorrelationConfig) {
    this.maxLag = config.maxLag || Math.floor(config.sampleRate / 50); // up to 20 Hz
  }
  /**
   * Compute the autocorrelation function for a PCM frame.
   * @param pcm - Input PCM audio frame
   * @returns Autocorrelation array (length = maxLag)
   */
  analyzeFrame(pcm: Float32Array): Float32Array {
    const result = new Float32Array(this.maxLag);
    for (let lag = 0; lag < this.maxLag; lag++) {
      let sum = 0;
      for (let i = 0; i < pcm.length - lag; i++) {
        sum += pcm[i] * pcm[i + lag];
      }
      result[lag] = sum / (pcm.length - lag);
    }
    return result;
  }
}
