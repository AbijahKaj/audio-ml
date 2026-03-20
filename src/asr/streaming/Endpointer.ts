import { VAD, type VADConfig } from '../../applications/speech/VAD.js';

export type EndpointEvent = 'speech' | 'silence' | 'speech-end';

export interface EndpointerConfig extends VADConfig {
  /** Minimum speech duration before declaring a valid utterance (ms) */
  minSpeechDurationMs?: number;
  /** Silence duration before triggering speech-end (ms) */
  endpointSilenceDurationMs?: number;
}

/**
 * Endpointer — wraps VAD to detect utterance boundaries for streaming ASR.
 *
 * Emits 'speech-end' after a configurable silence window following speech,
 * gating the ASR engine to finalize the current utterance.
 */
export class Endpointer {
  private readonly vad: VAD;
  private speechStarted = false;
  private speechFrames = 0;
  private silenceFramesAfterSpeech = 0;
  private readonly minSpeechFrames: number;
  private readonly endpointSilenceFrames: number;

  constructor(config: EndpointerConfig) {
    const frameMs = (config.sampleRate > 0)
      ? 1000 / (config.sampleRate / (config.fftSize ?? 1024))
      : 10;

    this.minSpeechFrames = Math.ceil(
      (config.minSpeechDurationMs ?? 300) / frameMs,
    );
    this.endpointSilenceFrames = Math.ceil(
      (config.endpointSilenceDurationMs ?? 800) / frameMs,
    );

    this.vad = new VAD(config);
  }

  processFrame(pcm: Float32Array): EndpointEvent {
    const result = this.vad.processFrame(pcm);

    if (result.isSpeech) {
      this.speechStarted = true;
      this.speechFrames++;
      this.silenceFramesAfterSpeech = 0;
      return 'speech';
    }

    if (this.speechStarted) {
      this.silenceFramesAfterSpeech++;

      if (
        this.speechFrames >= this.minSpeechFrames &&
        this.silenceFramesAfterSpeech >= this.endpointSilenceFrames
      ) {
        this.reset();
        return 'speech-end';
      }
    }

    return 'silence';
  }

  reset(): void {
    this.speechStarted = false;
    this.speechFrames = 0;
    this.silenceFramesAfterSpeech = 0;
    this.vad.reset();
  }

  get isSpeaking(): boolean {
    return this.speechStarted;
  }
}
