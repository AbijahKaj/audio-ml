/**
 * Simple EventEmitter implementation for browser and Node.js compatibility
 */
class SimpleEventEmitter {
  private events: Map<string, Array<(...args: any[]) => void>> = new Map();

  on(event: string, listener: (...args: any[]) => void): this {
    if (!this.events.has(event)) {
      this.events.set(event, []);
    }
    this.events.get(event)!.push(listener);
    return this;
  }

  off(event: string, listener: (...args: any[]) => void): this {
    const listeners = this.events.get(event);
    if (listeners) {
      const index = listeners.indexOf(listener);
      if (index > -1) {
        listeners.splice(index, 1);
      }
    }
    return this;
  }

  emit(event: string, ...args: any[]): boolean {
    const listeners = this.events.get(event);
    if (listeners && listeners.length > 0) {
      listeners.forEach(listener => listener(...args));
      return true;
    }
    return false;
  }

  once(event: string, listener: (...args: any[]) => void): this {
    const onceWrapper = (...args: any[]) => {
      listener(...args);
      this.off(event, onceWrapper);
    };
    return this.on(event, onceWrapper);
  }

  removeAllListeners(event?: string): this {
    if (event) {
      this.events.delete(event);
    } else {
      this.events.clear();
    }
    return this;
  }
}

/**
 * Base configuration interface for all applications
 */
export interface ApplicationConfig {
  sampleRate: number;
  [key: string]: any;
}

/**
 * Base class for all audio applications
 * Provides EventEmitter functionality and common lifecycle methods
 */
export abstract class BaseApplication extends SimpleEventEmitter {
  protected sampleRate: number;
  protected isProcessing: boolean = false;

  constructor(config: ApplicationConfig) {
    super();
    this.sampleRate = config.sampleRate;
  }

  /**
   * Process a single audio frame
   * @param pcm - Input PCM audio frame
   * @returns Application-specific output
   */
  abstract processFrame(pcm: Float32Array): any;

  /**
   * Start processing
   */
  start(): void {
    this.isProcessing = true;
    this.emit('start');
  }

  /**
   * Stop processing
   */
  stop(): void {
    this.isProcessing = false;
    this.emit('stop');
  }

  /**
   * Reset application state
   */
  reset(): void {
    this.emit('reset');
  }

  /**
   * Get current processing state
   */
  getProcessingState(): boolean {
    return this.isProcessing;
  }
}
