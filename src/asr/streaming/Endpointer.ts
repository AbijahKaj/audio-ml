import { VAD, type VADConfig } from '../../applications/speech/VAD';

export type EndpointDecision = 'speech' | 'silence' | 'speech-end';

export interface EndpointerConfig {
  sampleRate: number;
  vad?: Omit<VADConfig, 'sampleRate'>;
}

export class Endpointer {
  private vad: VAD;
  private wasSpeech = false;

  constructor(config: EndpointerConfig) {
    this.vad = new VAD({
      sampleRate: config.sampleRate,
      ...(config.vad ?? {}),
    });
  }

  processFrame(pcm: Float32Array): EndpointDecision {
    const result = this.vad.processFrame(pcm);
    const nowSpeech = result.isSpeech;

    if (this.wasSpeech && !nowSpeech) {
      this.wasSpeech = nowSpeech;
      return 'speech-end';
    }

    this.wasSpeech = nowSpeech;
    return nowSpeech ? 'speech' : 'silence';
  }

  reset(): void {
    this.vad.reset();
    this.wasSpeech = false;
  }
}
