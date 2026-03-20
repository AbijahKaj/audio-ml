import { VAD } from '../../applications/speech/VAD';

export type EndpointEvent = 'speech' | 'silence' | 'speech-end';

/**
 * Bridges VAD to utterance boundaries for streaming ASR.
 */
export class Endpointer {
  private readonly vad: VAD;

  constructor(sampleRate: number) {
    this.vad = new VAD({ sampleRate });
  }

  processFrame(pcm: Float32Array): EndpointEvent {
    const r = this.vad.processFrame(pcm);
    if (r.isSpeech) {
      return 'speech';
    }
    return 'silence';
  }
}