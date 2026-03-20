export type DecoderType = 'rnnt' | 'tdt';

export type SelfAttentionModel = 'rel_pos' | 'rel_pos_local_attn' | 'abs_pos';

export interface FastConformerConfig {
  encoderLayers: number;
  dModel: number;
  numHeads: number;
  convKernelSize: number;
  ffExpansionFactor: number;
  subsamplingFactor: number;
  vocabSize: number;
  predHidden: number;
  encHidden: number;
  jointHidden: number;
  jointActivation: 'relu' | 'tanh' | 'sigmoid';
  numMelBands: number;
  sampleRate: number;
  windowSizeMs: number;
  hopSizeMs: number;
  attContextSize: [number, number];
  decoderType: DecoderType;
  tdtNumDurations?: number;
  tdtDurations?: number[];
  selfAttentionModel: SelfAttentionModel;
  posEmbMaxLen: number;
  xscale: boolean;
  subsampling: string;
  subsamplingConvChannels: number;
  blankTokenId: number;
}

export function parseModelConfig(json: string): FastConformerConfig {
  const raw = JSON.parse(json) as Record<string, unknown>;
  const att = raw.att_context_size as [number, number] | number[] | undefined;
  const attTuple: [number, number] =
    att && att.length >= 2 ? [Number(att[0]), Number(att[1])] : [-1, -1];
  const decType = (raw.decoder_type as string) ?? 'rnnt';
  const tdtDurations = raw.tdt_durations as number[] | undefined;
  return {
    encoderLayers: Number(raw.encoder_layers),
    dModel: Number(raw.d_model),
    numHeads: Number(raw.num_heads),
    convKernelSize: Number(raw.conv_kernel_size),
    ffExpansionFactor: Number(raw.ff_expansion_factor),
    subsamplingFactor: Number(raw.subsampling_factor),
    vocabSize: Number(raw.vocab_size),
    predHidden: Number(raw.pred_hidden),
    encHidden: Number(raw.enc_hidden ?? raw.d_model),
    jointHidden: Number(raw.joint_hidden),
    jointActivation: (raw.joint_activation as FastConformerConfig['jointActivation']) ?? 'relu',
    numMelBands: Number(raw.num_mel_bands),
    sampleRate: Number(raw.sample_rate),
    windowSizeMs: Number(raw.window_size_ms),
    hopSizeMs: Number(raw.hop_size_ms),
    attContextSize: attTuple,
    decoderType: decType === 'tdt' ? 'tdt' : 'rnnt',
    tdtNumDurations: raw.tdt_num_durations != null ? Number(raw.tdt_num_durations) : undefined,
    tdtDurations: Array.isArray(tdtDurations) ? tdtDurations.map(Number) : undefined,
    selfAttentionModel:
      (raw.self_attention_model as SelfAttentionModel) ?? 'rel_pos_local_attn',
    posEmbMaxLen: Number(raw.pos_emb_max_len ?? 5000),
    xscale: raw.xscale !== false,
    subsampling: String(raw.subsampling ?? 'striding'),
    subsamplingConvChannels: Number(raw.subsampling_conv_channels ?? raw.d_model),
    blankTokenId: Number(raw.blank_token_id ?? raw.vocab_size ?? 0),
  };
}
