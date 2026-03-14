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
  private workletNode: AudioWorkletNode | null = null;
  private stream: MediaStream | null = null;
  private sourceNode: MediaElementAudioSourceNode | null = null;
  private audioElement: HTMLAudioElement | null = null;
  private listeners: Map<string, Set<Function>> = new Map();
  private sampleRate: number = 44100;
  private frameBuffer: Float32Array = new Float32Array(0);
  private targetFrameSize: number = 1024;

  constructor(sampleRate: number = 44100) {
    this.sampleRate = sampleRate;
  }

  private getWorkletUrl(): URL {
    return new URL('./pcm-processor.js', import.meta.url);
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
      await this.audioCtx.audioWorklet.addModule(this.getWorkletUrl());

      const source = this.audioCtx.createMediaStreamSource(this.stream);

      // Create AudioWorkletNode instead of ScriptProcessorNode
      this.workletNode = new AudioWorkletNode(this.audioCtx, 'pcm-processor');
      
      // Set up message handler to receive PCM data
      this.workletNode.port.onmessage = (event) => {
        if (event.data.type === 'pcm-data' && this.isActive) {
          const pcm = new Float32Array(event.data.data);
          this.bufferAndEmitFrames(pcm, event.data.sampleRate || this.sampleRate);
        }
      };

      source.connect(this.workletNode);
      this.workletNode.connect(this.audioCtx.destination);

      // Start the processor
      this.workletNode.port.postMessage({ type: 'start' });

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
      this.audioCtx = new AudioContext({ sampleRate: this.sampleRate });
      await this.audioCtx.audioWorklet.addModule(this.getWorkletUrl());

      // Create audio element for playback
      const audioUrl = URL.createObjectURL(file);
      this.audioElement = new Audio(audioUrl);
      this.audioElement.crossOrigin = 'anonymous';

      // Create source from audio element
      this.sourceNode = this.audioCtx.createMediaElementSource(this.audioElement);
      
      // Create AudioWorkletNode instead of ScriptProcessorNode
      this.workletNode = new AudioWorkletNode(this.audioCtx, 'pcm-processor');
      
      // Set up message handler to receive PCM data
      this.workletNode.port.onmessage = (event) => {
        if (event.data.type === 'pcm-data' && this.isActive && !this.audioElement?.paused) {
          const pcm = new Float32Array(event.data.data);
          this.bufferAndEmitFrames(pcm, event.data.sampleRate || this.audioCtx!.sampleRate);
        }
      };

      this.sourceNode.connect(this.workletNode);
      this.workletNode.connect(this.audioCtx.destination);

      // Start the processor
      this.workletNode.port.postMessage({ type: 'start' });

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

    // Emit any remaining buffered data
    if (this.frameBuffer.length > 0) {
      this.emit('pcm-data', new Float32Array(this.frameBuffer), this.sampleRate);
      this.frameBuffer = new Float32Array(0);
    }

    if (this.workletNode) {
      // Stop the processor
      this.workletNode.port.postMessage({ type: 'stop' });
      this.workletNode.disconnect();
      this.workletNode = null;
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

  /**
   * Buffer small frames and emit larger frames for analyzers
   */
  private bufferAndEmitFrames(pcm: Float32Array, sampleRate: number): void {
    // Append new data to buffer
    const newBuffer = new Float32Array(this.frameBuffer.length + pcm.length);
    newBuffer.set(this.frameBuffer, 0);
    newBuffer.set(pcm, this.frameBuffer.length);
    this.frameBuffer = newBuffer;

    // Emit frames when we have enough data
    while (this.frameBuffer.length >= this.targetFrameSize) {
      const frame = this.frameBuffer.subarray(0, this.targetFrameSize);
      this.emit('pcm-data', new Float32Array(frame), sampleRate);
      
      // Keep remaining data in buffer
      this.frameBuffer = this.frameBuffer.subarray(this.targetFrameSize);
    }
  }

  /**
   * Set target frame size for buffering (default: 1024)
   */
  setTargetFrameSize(size: number): void {
    this.targetFrameSize = size;
    this.frameBuffer = new Float32Array(0); // Reset buffer when changing size
  }
}
