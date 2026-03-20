import type { TensorHandle } from '../compute/types';
import type { FastConformerEncoder } from '../encoder/FastConformerEncoder';

/**
 * Runs the encoder on contiguous mel chunks. Full cache-aware attention is not wired here yet;
 * this class is the integration point for Phase 7 streaming.
 */
export class ChunkedInference {
  constructor(private readonly encoder: FastConformerEncoder) {}

  /** Encode one mel chunk [1, T, F]. Caller owns `mel` lifecycle. */
  encodeChunk(mel: TensorHandle): TensorHandle {
    return this.encoder.forward(mel);
  }
}