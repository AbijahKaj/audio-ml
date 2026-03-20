import { describe, expect, it } from 'vitest';
import { parseModelConfig } from '../src/asr/model/ModelConfig';
import { SentencePieceDecoder } from '../src/asr/text/SentencePieceDecoder';
import { Resampler } from '../src/asr/features/Resampler';

describe('ASR smoke tests', () => {
  it('parses model config with defaults', () => {
    const config = parseModelConfig({ decoder_type: 'rnnt', vocab_size: 1024 });
    expect(config.decoderType).toBe('rnnt');
    expect(config.vocabSize).toBe(1024);
    expect(config.numMelBands).toBe(80);
    expect(config.blankId).toBe(0);
  });

  it('decodes sentencepiece ids into normalized text', () => {
    const decoder = new SentencePieceDecoder(['<blank>', '▁hello', '▁world', '!']);
    expect(decoder.decode([1, 2, 3])).toBe('hello world!');
  });

  it('resamples to target sample rate', () => {
    const resampler = new Resampler(48000, 16000);
    const input = new Float32Array(480);
    for (let i = 0; i < input.length; i++) {
      input[i] = Math.sin((2 * Math.PI * i) / 64);
    }
    const output = resampler.resample(input);
    expect(output.length).toBeCloseTo(160, 0);
  });
});
