export type DecoderType = 'rnnt' | 'tdt';

export interface FastConformerConfig {
  encoderLayers: number;
  dModel: number;
  numHeads: number;
  convKernelSize: number;
  ffExpansionFactor: number;
  subsamplingFactor: number;
  vocabSize: number;
  predHidden: number;
  predNumLayers: number;
  jointDim: number;
  numMelBands: number;
  sampleRate: number;
  windowSizeMs: number;
  hopSizeMs: number;
  attContextSize: [number, number];
  decoderType: DecoderType;
  tdtDurations?: number[];
  subsamplingConvChannels: number;
  normalize: 'per_feature' | 'NA';
}

export function parseModelConfig(json: string): FastConformerConfig {
  const raw = JSON.parse(json);
  return {
    encoderLayers: raw.encoder_layers ?? 17,
    dModel: raw.d_model ?? 512,
    numHeads: raw.num_heads ?? 8,
    convKernelSize: raw.conv_kernel_size ?? 9,
    ffExpansionFactor: raw.ff_expansion_factor ?? 4,
    subsamplingFactor: raw.subsampling_factor ?? 8,
    vocabSize: raw.vocab_size ?? 1025,
    predHidden: raw.pred_hidden ?? 640,
    predNumLayers: raw.pred_num_layers ?? 1,
    jointDim: raw.joint_dim ?? 640,
    numMelBands: raw.num_mel_bands ?? 80,
    sampleRate: raw.sample_rate ?? 16000,
    windowSizeMs: raw.window_size_ms ?? 25,
    hopSizeMs: raw.hop_size_ms ?? 10,
    attContextSize: raw.att_context_size ?? [70, 1],
    decoderType: raw.decoder_type ?? 'rnnt',
    tdtDurations: raw.tdt_num_durations,
    subsamplingConvChannels: raw.subsampling_conv_channels ?? 256,
    normalize: raw.normalize ?? 'per_feature',
  };
}
