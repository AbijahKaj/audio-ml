/**
 * SentencePiece detokenizer — converts token IDs to text.
 *
 * Works with the vocab JSON exported by tools/export_vocab.py.
 * The vocab is a plain array: vocab[tokenId] = tokenString.
 *
 * SentencePiece uses '▁' (U+2581 LOWER ONE EIGHTH BLOCK) as the word boundary
 * marker. After joining all tokens, we replace '▁' with a space.
 *
 * Token 0 is always blank; blank tokens are skipped.
 */
export class SentencePieceDecoder {
  private readonly vocab: string[];
  private readonly blankId: number;

  constructor(vocabJson: string, blankId = 0) {
    this.vocab = JSON.parse(vocabJson) as string[];
    this.blankId = blankId;
  }

  decode(tokenIds: number[]): string {
    return tokenIds
      .filter(id => id !== this.blankId)
      .map(id => this.vocab[id] ?? '')
      .join('')
      .replace(/▁/g, ' ')
      .trim();
  }

  /** Decode and return partial text (for streaming display). */
  decodePartial(tokenIds: number[]): string {
    return this.decode(tokenIds);
  }

  get vocabSize(): number {
    return this.vocab.length;
  }
}
