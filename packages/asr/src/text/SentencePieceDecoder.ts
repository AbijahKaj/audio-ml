/**
 * SentencePiece detokenizer.
 * Converts token IDs back to text using a vocabulary lookup table.
 *
 * Handles:
 *  - The ▁ (U+2581) word boundary marker used by SentencePiece.
 *  - Byte-fallback tokens of the form `<0xNN>`, emitted by multilingual
 *    unigram tokenizers (e.g. Parakeet TDT 0.6B v3) for characters that are
 *    not in the subword vocabulary. Consecutive byte tokens form a single
 *    UTF-8 sequence and are decoded together.
 *  - SentencePiece control/special tokens (`<blank>`, `<unk>`, `<pad>`,
 *    `<s>`, `</s>`, `<eos>`, `<bos>`), which are dropped from the output.
 */
const BYTE_TOKEN_RE = /^<0x([0-9A-Fa-f]{2})>$/;

const SPECIAL_TOKENS = new Set([
  '<blank>',
  '<unk>',
  '<pad>',
  '<s>',
  '</s>',
  '<bos>',
  '<eos>',
  '<sep>',
  '<cls>',
  '<mask>',
]);

const utf8Decoder =
  typeof TextDecoder !== 'undefined'
    ? new TextDecoder('utf-8', { fatal: false })
    : null;

export class SentencePieceDecoder {
  private vocab: string[];

  constructor(vocabJson: string) {
    const parsed = JSON.parse(vocabJson);
    if (Array.isArray(parsed)) {
      this.vocab = parsed;
    } else if (parsed.tokens) {
      this.vocab = parsed.tokens;
    } else {
      // Object form ({ "0": "<piece>", ... }). Order entries by their numeric
      // id so that token-id lookups map to the correct piece.
      const entries = Object.entries(parsed) as [string, string][];
      const allNumeric = entries.every(([k]) => /^\d+$/.test(k));
      if (allNumeric) {
        entries.sort((a, b) => Number(a[0]) - Number(b[0]));
      }
      this.vocab = entries.map(([, v]) => v);
    }
  }

  decode(tokenIds: number[]): string {
    const pieces: string[] = [];
    let byteBuffer: number[] = [];

    const flushBytes = () => {
      if (byteBuffer.length === 0) return;
      pieces.push(this.bytesToString(byteBuffer));
      byteBuffer = [];
    };

    for (const id of tokenIds) {
      if (id < 0 || id >= this.vocab.length) {
        flushBytes();
        continue;
      }
      const piece = this.vocab[id] ?? '';

      const byteMatch = BYTE_TOKEN_RE.exec(piece);
      if (byteMatch) {
        byteBuffer.push(parseInt(byteMatch[1], 16));
        continue;
      }

      flushBytes();

      if (SPECIAL_TOKENS.has(piece)) continue;
      pieces.push(piece);
    }

    flushBytes();

    return pieces.join('').replace(/▁/g, ' ').trim();
  }

  private bytesToString(bytes: number[]): string {
    const arr = Uint8Array.from(bytes);
    if (utf8Decoder) return utf8Decoder.decode(arr);
    // Fallback for environments without TextDecoder.
    let out = '';
    for (const b of bytes) out += String.fromCharCode(b);
    return out;
  }

  get vocabSize(): number {
    return this.vocab.length;
  }

  getToken(id: number): string {
    if (id < 0 || id >= this.vocab.length) return '';
    return this.vocab[id] ?? '';
  }
}
