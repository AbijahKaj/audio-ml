export interface StreamingAsrHost {
  transcribeBuffer(audio: Float32Array): Promise<{ text: string; latencyMs: number }>;
  emit(event: string, payload: unknown): boolean;
}

/**
 * Buffers PCM frames and drives partial / final callbacks.
 * Full encoder KV-cache streaming is not yet wired; partials re-run the full encoder on the
 * accumulated buffer (higher CPU use, correct for short utterances).
 */
export class ChunkedInference {
  private buffer: Float32Array = new Float32Array(0);
  private partialIntervalFrames: number;
  private frameSamples: number;
  private sincePartial = 0;

  constructor(
    private readonly host: StreamingAsrHost,
    opts?: { partialEveryNFrames?: number; frameSamples?: number },
  ) {
    this.partialIntervalFrames = opts?.partialEveryNFrames ?? 10;
    this.frameSamples = opts?.frameSamples ?? 512;
  }

  reset(): void {
    this.buffer = new Float32Array(0);
    this.sincePartial = 0;
  }

  pushPcm(frame: Float32Array): void {
    const merged = new Float32Array(this.buffer.length + frame.length);
    merged.set(this.buffer);
    merged.set(frame, this.buffer.length);
    this.buffer = merged;
    this.sincePartial++;
    if (this.sincePartial >= this.partialIntervalFrames && this.buffer.length >= this.frameSamples) {
      this.sincePartial = 0;
      void this.emitPartial();
    }
  }

  private async emitPartial(): Promise<void> {
    try {
      const result = await this.host.transcribeBuffer(this.buffer);
      this.host.emit('partial', { text: result.text, latencyMs: result.latencyMs });
    } catch {
      /* model not ready or OOM on long buffer — skip partial */
    }
  }

  async finalize(): Promise<void> {
    if (this.buffer.length === 0) {
      return;
    }
    const result = await this.host.transcribeBuffer(this.buffer);
    this.host.emit('final', result);
    this.reset();
  }
}
