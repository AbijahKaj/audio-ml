export interface ChunkedInferenceConfig {
  sampleRate: number;
  chunkDurationMs?: number;
}

function concatFloat32(a: Float32Array, b: Float32Array): Float32Array {
  const output = new Float32Array(a.length + b.length);
  output.set(a, 0);
  output.set(b, a.length);
  return output;
}

export class ChunkedInference {
  private readonly chunkSizeSamples: number;
  private pending: Float32Array = new Float32Array(0);

  constructor(config: ChunkedInferenceConfig) {
    this.chunkSizeSamples = Math.max(
      1,
      Math.round(config.sampleRate * (config.chunkDurationMs ?? 160) / 1000),
    );
  }

  push(audio: Float32Array): Float32Array[] {
    this.pending = concatFloat32(this.pending, audio);
    const chunks: Float32Array[] = [];

    while (this.pending.length >= this.chunkSizeSamples) {
      chunks.push(Float32Array.from(this.pending.slice(0, this.chunkSizeSamples)));
      this.pending = Float32Array.from(this.pending.slice(this.chunkSizeSamples));
    }

    return chunks;
  }

  flush(): Float32Array | null {
    if (this.pending.length === 0) {
      return null;
    }

    const finalChunk = this.pending;
    this.pending = new Float32Array(0);
    return finalChunk;
  }

  reset(): void {
    this.pending = new Float32Array(0);
  }
}
