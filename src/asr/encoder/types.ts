import type { TensorHandle } from '../compute/types';

export interface AttentionCache {
  k: TensorHandle;
  v: TensorHandle;
}

export interface ConformerLayerCache {
  attention?: AttentionCache;
  convState?: TensorHandle;
}
