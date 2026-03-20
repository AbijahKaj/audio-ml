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
  numMelBands: number;
  sampleRate: number;
  windowSizeMs: number;
  hopSizeMs: number;
  attContextSize: [number, number];
  decoderType: DecoderType;
  tdtNumDurations?: number;
  tdtDurations?: number[];
  blankId: number;
}

interface RawFastConformerConfig {
  encoder_layers?: number;
  d_model?: number;
  num_heads?: number;
  conv_kernel_size?: number;
  ff_expansion_factor?: number;
  subsampling_factor?: number;
  vocab_size?: number;
  pred_hidden?: number;
  num_mel_bands?: number;
  sample_rate?: number;
  window_size_ms?: number;
  hop_size_ms?: number;
  att_context_size?: [number, number];
  decoder_type?: DecoderType;
  tdt_num_durations?: number | number[];
  tdt_durations?: number[];
  blank_id?: number;
}

function expectNumber(value: unknown, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

export function parseModelConfig(source: string | RawFastConformerConfig): FastConformerConfig {
  const raw = typeof source === 'string' ? (JSON.parse(source) as RawFastConformerConfig) : source;
  const decoderType: DecoderType = raw.decoder_type === 'tdt' ? 'tdt' : 'rnnt';
  const tdtDurations = Array.isArray(raw.tdt_durations)
    ? raw.tdt_durations
    : Array.isArray(raw.tdt_num_durations)
      ? raw.tdt_num_durations
      : undefined;

  const config: FastConformerConfig = {
    encoderLayers: expectNumber(raw.encoder_layers, 17),
    dModel: expectNumber(raw.d_model, 512),
    numHeads: expectNumber(raw.num_heads, 8),
    convKernelSize: expectNumber(raw.conv_kernel_size, 9),
    ffExpansionFactor: expectNumber(raw.ff_expansion_factor, 4),
    subsamplingFactor: expectNumber(raw.subsampling_factor, 8),
    vocabSize: expectNumber(raw.vocab_size, 1024),
    predHidden: expectNumber(raw.pred_hidden, 640),
    numMelBands: expectNumber(raw.num_mel_bands, 80),
    sampleRate: expectNumber(raw.sample_rate, 16000),
    windowSizeMs: expectNumber(raw.window_size_ms, 25),
    hopSizeMs: expectNumber(raw.hop_size_ms, 10),
    attContextSize: Array.isArray(raw.att_context_size) && raw.att_context_size.length === 2
      ? raw.att_context_size
      : [70, 1],
    decoderType,
    tdtNumDurations: typeof raw.tdt_num_durations === 'number' ? raw.tdt_num_durations : tdtDurations?.length,
    tdtDurations,
    blankId: expectNumber(raw.blank_id, 0),
  };

  if (config.decoderType === 'tdt' && !config.tdtDurations) {
    const bins = config.tdtNumDurations ?? 5;
    config.tdtDurations = Array.from({ length: bins }, (_, index) => index);
  }

  return config;
}
