import { VAD, type VADConfig } from '../../applications/speech/VAD';

export type EndpointEvent = 'speech' | 'silence' | 'speech-end';

export interface EndpointerConfig {
  sampleRate: number;
  fftSize?: number;
  silenceTimeoutMs?: number;
  frameSize?: number;
}

/**
 * Endpointer wraps VAD to detect utterance boundaries for streaming ASR.
 * Tracks consecutive silence frames and signals when an utterance has ended.
 */
export class Endpointer {
  private vad: VAD;
  private silenceTimeoutFrames: number;
  private consecutiveSilence: number = 0;
  private hasSpeechStarted: boolean = false;
  private frameSize: number;
  private sampleRate: number;

  constructor(config: EndpointerConfig) {
    this.sampleRate = config.sampleRate;
    this.frameSize = config.frameSize ?? 1024;

    const vadConfig: VADConfig = {
      sampleRate: config.sampleRate,
      fftSize: config.fftSize ?? 1024,
      speechFramesRequired: 2,
      silenceFramesRequired: 3,
    };
    this.vad = new VAD(vadConfig);

    const silenceTimeoutMs = config.silenceTimeoutMs ?? 800;
    const frameDurationMs = (this.frameSize / this.sampleRate) * 1000;
    this.silenceTimeoutFrames = Math.ceil(silenceTimeoutMs / frameDurationMs);
  }

  processFrame(pcm: Float32Array): EndpointEvent {
    const result = this.vad.processFrame(pcm);

    if (result.isSpeech) {
      this.hasSpeechStarted = true;
      this.consecutiveSilence = 0;
      return 'speech';
    }

    this.consecutiveSilence++;

    if (this.hasSpeechStarted && this.consecutiveSilence >= this.silenceTimeoutFrames) {
      this.hasSpeechStarted = false;
      this.consecutiveSilence = 0;
      return 'speech-end';
    }

    return 'silence';
  }

  reset(): void {
    this.consecutiveSilence = 0;
    this.hasSpeechStarted = false;
  }

  get isSpeechActive(): boolean {
    return this.hasSpeechStarted;
  }
}
