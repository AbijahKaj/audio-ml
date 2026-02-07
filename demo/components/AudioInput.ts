/**
 * Audio Input Component
 * Handles both microphone recording and file upload/playback
 * Emits PCM data events for processing
 */

export type AudioInputMode = 'microphone' | 'file';

export interface AudioInputEvents {
  'pcm-data': (pcm: Float32Array, sampleRate: number) => void;
  'start': () => void;
  'stop': () => void;
  'error': (error: Error) => void;
}

export class AudioInput {
  private mode: AudioInputMode = 'microphone';
  private isActive: boolean = false;
  private audioCtx: AudioContext | null = null;
  private processor: ScriptProcessorNode | null = null;
  private stream: MediaStream | null = null;
  private sourceNode: MediaElementAudioSourceNode | null = null;
  private audioElement: HTMLAudioElement | null = null;
  private listeners: Map<string, Set<Function>> = new Map();
  private sampleRate: number = 44100;
  private bufferSize: number = 4096;

  constructor(sampleRate: number = 44100, bufferSize: number = 4096) {
    this.sampleRate = sampleRate;
    this.bufferSize = bufferSize;
  }

  /**
   * Add event listener
   */
  on<K extends keyof AudioInputEvents>(event: K, listener: AudioInputEvents[K]): this {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, new Set());
    }
    this.listeners.get(event)!.add(listener);
    return this;
  }

  /**
   * Remove event listener
   */
  off<K extends keyof AudioInputEvents>(event: K, listener: AudioInputEvents[K]): this {
    const listeners = this.listeners.get(event);
    if (listeners) {
      listeners.delete(listener);
    }
    return this;
  }

  /**
   * Emit event
   */
  private emit<K extends keyof AudioInputEvents>(event: K, ...args: Parameters<AudioInputEvents[K]>): void {
    const listeners = this.listeners.get(event);
    if (listeners) {
      listeners.forEach(listener => {
        try {
          (listener as any)(...args);
        } catch (error) {
          console.error(`Error in ${event} listener:`, error);
        }
      });
    }
  }

  /**
   * Start microphone recording
   */
  async startMicrophone(): Promise<void> {
    if (this.isActive) {
      await this.stop();
    }

    try {
      this.mode = 'microphone';
      this.stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          sampleRate: this.sampleRate,
          echoCancellation: false,
          noiseSuppression: false,
        },
      });

      this.audioCtx = new AudioContext({ sampleRate: this.sampleRate });
      const source = this.audioCtx.createMediaStreamSource(this.stream);

      this.processor = this.audioCtx.createScriptProcessor(this.bufferSize, 1, 1);
      source.connect(this.processor);
      this.processor.connect(this.audioCtx.destination);

      this.processor.onaudioprocess = (event) => {
        if (this.isActive) {
          const input = event.inputBuffer.getChannelData(0);
          // Create a copy to avoid issues with buffer reuse
          const pcm = new Float32Array(input);
          this.emit('pcm-data', pcm, this.sampleRate);
        }
      };

      this.isActive = true;
      this.emit('start');
    } catch (error) {
      const err = error instanceof Error ? error : new Error(String(error));
      this.emit('error', err);
      throw err;
    }
  }

  /**
   * Load and play audio file
   */
  async loadFile(file: File): Promise<void> {
    if (this.isActive) {
      await this.stop();
    }

    try {
      this.mode = 'file';
      
      // Decode audio file
      const arrayBuffer = await file.arrayBuffer();
      this.audioCtx = new AudioContext({ sampleRate: this.sampleRate });
      const audioBuffer = await this.audioCtx.decodeAudioData(arrayBuffer);

      // Create audio element for playback
      const audioUrl = URL.createObjectURL(file);
      this.audioElement = new Audio(audioUrl);
      this.audioElement.crossOrigin = 'anonymous';

      // Create source from audio element
      this.sourceNode = this.audioCtx.createMediaElementSource(this.audioElement);
      
      // Create processor to capture PCM data
      this.processor = this.audioCtx.createScriptProcessor(this.bufferSize, 1, 1);
      this.sourceNode.connect(this.processor);
      this.processor.connect(this.audioCtx.destination);

      // Handle PCM data
      this.processor.onaudioprocess = (event) => {
        if (this.isActive && !this.audioElement?.paused) {
          const input = event.inputBuffer.getChannelData(0);
          const pcm = new Float32Array(input);
          this.emit('pcm-data', pcm, this.audioCtx!.sampleRate);
        }
      };

      // Handle playback end
      this.audioElement.addEventListener('ended', () => {
        this.stop();
      });

      // Start playback
      this.isActive = true;
      await this.audioElement.play();
      this.emit('start');
    } catch (error) {
      const err = error instanceof Error ? error : new Error(String(error));
      this.emit('error', err);
      throw err;
    }
  }

  /**
   * Stop audio input
   */
  async stop(): Promise<void> {
    this.isActive = false;

    if (this.processor) {
      this.processor.disconnect();
      this.processor = null;
    }

    if (this.mode === 'microphone') {
      if (this.stream) {
        this.stream.getTracks().forEach(track => track.stop());
        this.stream = null;
      }
    } else if (this.mode === 'file') {
      if (this.audioElement) {
        this.audioElement.pause();
        this.audioElement.currentTime = 0;
        this.audioElement = null;
      }
      if (this.sourceNode) {
        this.sourceNode.disconnect();
        this.sourceNode = null;
      }
    }

    if (this.audioCtx) {
      await this.audioCtx.close();
      this.audioCtx = null;
    }

    this.emit('stop');
  }

  /**
   * Get current mode
   */
  getMode(): AudioInputMode {
    return this.mode;
  }

  /**
   * Check if active
   */
  isRunning(): boolean {
    return this.isActive;
  }

  /**
   * Get current sample rate
   */
  getSampleRate(): number {
    return this.audioCtx?.sampleRate || this.sampleRate;
  }
}
