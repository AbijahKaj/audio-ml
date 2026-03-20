import type { TensorHandle } from '../compute/types';

export interface PredictionState {
  h: TensorHandle;
  c: TensorHandle;
  lastToken: number;
}

export interface DecoderState extends PredictionState {
  frameOffset: number;
}

export interface DecodeResult {
  tokenIds: number[];
  state: DecoderState;
  framesConsumed: number;
}
