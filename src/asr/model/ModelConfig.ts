/**
 * Model configuration for FastConformer ASR models.
 * Parsed from model_config.json exported by tools/export_nemo_to_safetensors.py.
 */

export type DecoderType = 'rnnt' | 'tdt';

export interface FastConformerConfig {
  /** Number of ConformerBlock layers (17 for Parakeet 120M, 24 for Nemotron 0.6B) */
  encoderLayers: number;
  /** Model dimension — 512 for most models */
  dModel: number;
  /** Number of attention heads — 8 */
  numHeads: number;
  /** Depthwise conv kernel size in the conv module — 9 */
  convKernelSize: number;
  /** FFN expansion factor — 4 (hidden dim = dModel * 4) */
  ffExpansionFactor: number;
  /** Total time-domain subsampling factor — 8 (two stride-2 conv2d layers) */
  subsamplingFactor: number;
  /** Number of channels in the subsampling conv layers — 256 */
  subsamplingConvChannels: number;
  /** Vocabulary size (including blank token at index 0) */
  vocabSize: number;
  /** Prediction network LSTM hidden size */
  predHidden: number;
  /** Prediction network embedding dimension */
  predEmbedDim: number;
  /** Number of log-mel bands — 80 */
  numMelBands: number;
  /** Input sample rate — 16000 Hz */
  sampleRate: number;
  /** Analysis window size in ms — 25 */
  windowSizeMs: number;
  /** Hop size between frames in ms — 10 */
  hopSizeMs: number;
  /**
   * Attention context size [left_context, right_context].
   * [70, 1] for streaming models — limits attention to 70 past + 1 future frames.
   * [-1, -1] for full-context (offline) models.
   */
  attContextSize: [number, number];
  /** Which decoder variant: 'rnnt' (standard) or 'tdt' (token-and-duration) */
  decoderType: DecoderType;
  /**
   * TDT only: duration bins.
   * e.g. [0, 1, 2, 3, 4] — the duration vocab maps index → frames to skip.
   */
  tdtDurations?: number[];
  /** TDT only: number of duration bins */
  tdtNumDurations?: number;
}

interface RawConfig {
  encoder_layers: number;
  d_model: number;
  num_heads: number;
  conv_kernel_size: number;
  ff_expansion_factor: number;
  subsampling_factor?: number;
  subsampling_conv_channels?: number;
  vocab_size: number;
  pred_hidden: number;
  pred_embed_dim?: number;
  num_mel_bands: number;
  sample_rate: number;
  window_size_ms: number;
  hop_size_ms: number;
  att_context_size?: [number, number];
  decoder_type?: string;
  tdt_durations?: number[];
  tdt_num_durations?: number;
}

export function parseModelConfig(json: string): FastConformerConfig {
  const raw = JSON.parse(json) as RawConfig;

  const decoderType: DecoderType =
    raw.decoder_type === 'tdt' ? 'tdt' : 'rnnt';

  return {
    encoderLayers: raw.encoder_layers,
    dModel: raw.d_model,
    numHeads: raw.num_heads,
    convKernelSize: raw.conv_kernel_size,
    ffExpansionFactor: raw.ff_expansion_factor,
    subsamplingFactor: raw.subsampling_factor ?? 8,
    subsamplingConvChannels: raw.subsampling_conv_channels ?? 256,
    vocabSize: raw.vocab_size,
    predHidden: raw.pred_hidden,
    predEmbedDim: raw.pred_embed_dim ?? 128,
    numMelBands: raw.num_mel_bands,
    sampleRate: raw.sample_rate,
    windowSizeMs: raw.window_size_ms,
    hopSizeMs: raw.hop_size_ms,
    attContextSize: raw.att_context_size ?? [-1, -1],
    decoderType,
    tdtDurations: raw.tdt_durations,
    tdtNumDurations: raw.tdt_num_durations,
  };
}
