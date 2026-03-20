import { VAD, type VADConfig } from '../../applications/speech/VAD';

export type EndpointState = 'speech' | 'silence' | 'speech-end';

export class Endpointer {
  private readonly vad: VAD;

  constructor(config: VADConfig) {
    this.vad = new VAD(config);
  }

  processFrame(pcm: Float32Array): EndpointState {
    const previousState = this.vad.getState().isSpeech;
    const result = this.vad.processFrame(pcm);

    if (previousState && !result.isSpeech) {
      return 'speech-end';
    }

    return result.isSpeech ? 'speech' : 'silence';
  }

  reset(): void {
    this.vad.reset();
  }
}
