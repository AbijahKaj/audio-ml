/**
 * RMSEAnalyzer computes the Root Mean Square Energy (RMSE) of a PCM audio frame.
 * RMSE is a simple measure of signal energy, useful for detecting silence, loudness, or dynamic range.
 * Usage: const rmse = new RMSEAnalyzer({ sampleRate: 16000 }); rmse.analyzeFrame(pcm)
 */
export interface RMSEConfig { sampleRate: number }

export class RMSEAnalyzer {
  /**
   * @param _config - Configuration with sample rate
   */
  constructor(_config: RMSEConfig) {
  }
  /**
   * Compute the root mean square energy for a PCM frame.
   * @param pcm - Input PCM audio frame
   * @returns RMSE value
   */
  analyzeFrame(pcm: Float32Array): number {
    let sum = 0;
    for (let i = 0; i < pcm.length; i++) {
      sum += pcm[i] * pcm[i];
    }
    return Math.sqrt(sum / pcm.length);
  }
}
