/**
 * SentencePiece detokenizer.
 * Converts token IDs back to text using a vocabulary lookup table.
 * Handles the ▁ (U+2581) word boundary marker used by SentencePiece.
 */
export class SentencePieceDecoder {
  private vocab: string[];

  constructor(vocabJson: string) {
    const parsed = JSON.parse(vocabJson);
    if (Array.isArray(parsed)) {
      this.vocab = parsed;
    } else if (parsed.tokens) {
      this.vocab = parsed.tokens;
    } else {
      this.vocab = Object.values(parsed) as string[];
    }
  }

  decode(tokenIds: number[]): string {
    return tokenIds
      .map(id => {
        if (id < 0 || id >= this.vocab.length) return '';
        return this.vocab[id] ?? '';
      })
      .join('')
      .replace(/▁/g, ' ')
      .trim();
  }

  get vocabSize(): number {
    return this.vocab.length;
  }

  getToken(id: number): string {
    if (id < 0 || id >= this.vocab.length) return '';
    return this.vocab[id] ?? '';
  }
}
