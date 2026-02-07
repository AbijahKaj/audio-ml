/**
 * WaveformEnvelopeAnalyzer computes the amplitude envelope of a PCM audio frame.
 * The envelope tracks the maximum absolute amplitude in a sliding window, useful for visualizing signal dynamics.
 * Usage: const env = new WaveformEnvelopeAnalyzer({ sampleRate: 16000 }); env.analyzeFrame(pcm)
 */
export interface WaveformEnvelopeConfig { sampleRate: number, windowSize?: number }

export class WaveformEnvelopeAnalyzer {
  private windowSize: number;
  /**
   * @param config - Configuration with sample rate and optional window size
   */
  constructor(config: WaveformEnvelopeConfig) {
    this.windowSize = config.windowSize || Math.floor(config.sampleRate * 0.01); // 10ms default
  }
  /**
   * Compute the amplitude envelope for a PCM frame.
   * @param pcm - Input PCM audio frame
   * @returns Envelope as a Float32Array
   */
  analyzeFrame(pcm: Float32Array): Float32Array {
    const env = new Float32Array(pcm.length);
    for (let i = 0; i < pcm.length; i++) {
      let start = Math.max(0, i - Math.floor(this.windowSize / 2));
      let end = Math.min(pcm.length, i + Math.floor(this.windowSize / 2));
      let max = 0;
      for (let j = start; j < end; j++) {
        max = Math.max(max, Math.abs(pcm[j]));
      }
      env[i] = max;
    }
    return env;
  }
}
