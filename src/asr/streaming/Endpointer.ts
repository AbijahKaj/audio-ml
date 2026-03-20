import { VAD } from '../../applications/speech/VAD';

/**
 * Maps VAD events to streaming endpoint decisions for {@link SpeechRecognizer}.
 */
export class Endpointer {
  private readonly vad: VAD;
  private speechActive = false;

  constructor(sampleRate: number) {
    this.vad = new VAD({ sampleRate });
    this.vad.on('speech-start', () => {
      this.speechActive = true;
    });
    this.vad.on('speech-end', () => {
      this.speechActive = false;
    });
  }

  processFrame(pcm: Float32Array): 'speech' | 'silence' | 'speech-end' {
    const prev = this.speechActive;
    this.vad.processFrame(pcm);
    if (prev && !this.speechActive) {
      return 'speech-end';
    }
    return this.speechActive ? 'speech' : 'silence';
  }

  reset(): void {
    this.vad.reset();
    this.speechActive = false;
  }
}
