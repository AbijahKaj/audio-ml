export type DecoderType = 'rnnt' | 'tdt';

export type SelfAttentionModel = 'rel_pos' | 'rel_pos_local_attn';

/**
 * Parsed from NeMo JSON (see export_nemo_to_safetensors.py in tools/).
 */
export interface FastConformerConfig {
  encoderLayers: number;
  dModel: number;
  numHeads: number;
  convKernelSize: number;
  ffExpansionFactor: number;
  subsamplingFactor: number;
  subsamplingConvChannels: number;
  vocabSize: number;
  predHidden: number;
  predNumLayers?: number;
  embedDim?: number;
  numMelBands: number;
  sampleRate: number;
  windowSizeMs: number;
  hopSizeMs: number;
  attContextSize: [number, number];
  decoderType: DecoderType;
  tdtNumDurations?: number;
  tdtDurations?: number[];
  /** Optional per-feature mean/std (length = numMelBands) for NeMo-style normalization */
  featureMean?: number[];
  featureStd?: number[];
  /** NeMo-style positional encoding scale (sqrt(d_model)) */
  xscale?: boolean;
  posEmbMaxLen?: number;
  selfAttentionModel?: SelfAttentionModel;
  globalTokens?: number;
  /** Prefix before `encoder.*` keys (e.g. `"model."` or `""`) */
  stateDictPrefix?: string;
  /** Prefix before `decoder.*` keys (default `"decoder."`) */
  decoderStateDictPrefix?: string;
  blankTokenId?: number;
}

export function parseModelConfig(json: string): FastConformerConfig {
  const raw = JSON.parse(json) as Record<string, unknown>;
  const decoderType = (raw.decoder_type as string) ?? 'rnnt';
  const tdt = raw.tdt_num_durations as number | undefined;
  const tdtDur = raw.tdt_durations as number[] | undefined;
  const dModel = num(raw.d_model, 512);

  return {
    encoderLayers: num(raw.encoder_layers, 17),
    dModel,
    numHeads: num(raw.num_heads, 8),
    convKernelSize: num(raw.conv_kernel_size, 9),
    ffExpansionFactor: num(raw.ff_expansion_factor, 4),
    subsamplingFactor: num(raw.subsampling_factor, 8),
    subsamplingConvChannels:
      raw.subsampling_conv_channels !== undefined ? num(raw.subsampling_conv_channels, 512) : dModel,
    vocabSize: num(raw.vocab_size, 1024),
    predHidden: num(raw.pred_hidden, 640),
    predNumLayers: raw.pred_num_layers !== undefined ? (raw.pred_num_layers as number) : 1,
    embedDim: raw.embed_dim !== undefined ? (raw.embed_dim as number) : undefined,
    numMelBands: num(raw.num_mel_bands, 80),
    sampleRate: num(raw.sample_rate, 16000),
    windowSizeMs: num(raw.window_size_ms, 25),
    hopSizeMs: num(raw.hop_size_ms, 10),
    attContextSize: pair(raw.att_context_size as [number, number] | undefined, [-1, -1]),
    decoderType: decoderType === 'tdt' ? 'tdt' : 'rnnt',
    tdtNumDurations: tdt,
    tdtDurations: tdtDur,
    featureMean: raw.feature_mean as number[] | undefined,
    featureStd: raw.feature_std as number[] | undefined,
    xscale: raw.xscale !== undefined ? Boolean(raw.xscale) : true,
    posEmbMaxLen: raw.pos_emb_max_len !== undefined ? (raw.pos_emb_max_len as number) : 5000,
    selfAttentionModel: (raw.self_attention_model as SelfAttentionModel) ?? 'rel_pos',
    globalTokens: raw.global_tokens !== undefined ? (raw.global_tokens as number) : 0,
    stateDictPrefix: (raw.state_dict_prefix as string) ?? '',
    decoderStateDictPrefix: (raw.decoder_state_dict_prefix as string) ?? '',
    blankTokenId: raw.blank_token_id !== undefined ? (raw.blank_token_id as number) : 0,
  };
}

function num(v: unknown, d: number): number {
  return typeof v === 'number' && !Number.isNaN(v) ? v : d;
}

function pair(v: [number, number] | undefined, d: [number, number]): [number, number] {
  if (v && v.length === 2) {
    return [v[0], v[1]];
  }
  return d;
}