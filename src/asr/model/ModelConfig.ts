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
}

function requiredNumber(value: unknown, field: string, fallback?: number): number {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return value;
  }
  if (typeof fallback === 'number') {
    return fallback;
  }
  throw new Error(`Invalid or missing model config field: ${field}`);
}

export function parseModelConfig(json: string): FastConformerConfig {
  const raw = JSON.parse(json) as Record<string, unknown>;
  const attContext = raw.att_context_size;
  const durations = Array.isArray(raw.tdt_durations)
    ? raw.tdt_durations.filter((value): value is number => typeof value === 'number')
    : undefined;

  return {
    encoderLayers: requiredNumber(raw.encoder_layers, 'encoder_layers'),
    dModel: requiredNumber(raw.d_model, 'd_model'),
    numHeads: requiredNumber(raw.num_heads, 'num_heads'),
    convKernelSize: requiredNumber(raw.conv_kernel_size, 'conv_kernel_size', 9),
    ffExpansionFactor: requiredNumber(raw.ff_expansion_factor, 'ff_expansion_factor', 4),
    subsamplingFactor: requiredNumber(raw.subsampling_factor, 'subsampling_factor', 8),
    vocabSize: requiredNumber(raw.vocab_size, 'vocab_size'),
    predHidden: requiredNumber(raw.pred_hidden, 'pred_hidden'),
    numMelBands: requiredNumber(raw.num_mel_bands, 'num_mel_bands', 80),
    sampleRate: requiredNumber(raw.sample_rate, 'sample_rate', 16000),
    windowSizeMs: requiredNumber(raw.window_size_ms, 'window_size_ms', 25),
    hopSizeMs: requiredNumber(raw.hop_size_ms, 'hop_size_ms', 10),
    attContextSize: Array.isArray(attContext) && attContext.length >= 2
      ? [requiredNumber(attContext[0], 'att_context_size[0]'), requiredNumber(attContext[1], 'att_context_size[1]')]
      : [70, 1],
    decoderType: raw.decoder_type === 'tdt' ? 'tdt' : 'rnnt',
    tdtNumDurations: typeof raw.tdt_num_durations === 'number' ? raw.tdt_num_durations : durations?.length,
    tdtDurations: durations,
  };
}
