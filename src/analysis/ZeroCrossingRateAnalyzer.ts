/**
 * ZeroCrossingRateAnalyzer computes the rate at which the audio signal changes sign (crosses zero).
 * This is a simple time-domain feature useful for distinguishing voiced/unvoiced speech and percussive sounds.
 * Usage: const zcr = new ZeroCrossingRateAnalyzer({ sampleRate: 16000 }); zcr.analyzeFrame(pcm)
 */
export interface ZeroCrossingRateConfig { sampleRate: number }

export class ZeroCrossingRateAnalyzer {
  /**
   * @param _config - Configuration with sample rate
   */
  constructor(_config: ZeroCrossingRateConfig) {
  }
  /**
   * Compute the zero crossing rate for a PCM frame.
   * @param pcm - Input PCM audio frame
   * @returns Zero crossing rate (crossings per sample)
   */
  analyzeFrame(pcm: Float32Array): number {
    let zcr = 0;
    for (let i = 1; i < pcm.length; i++) {
      if ((pcm[i - 1] >= 0 && pcm[i] < 0) || (pcm[i - 1] < 0 && pcm[i] >= 0)) {
        zcr++;
      }
    }
    return zcr / pcm.length;
  }
}
