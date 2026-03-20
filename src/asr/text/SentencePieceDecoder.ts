/**
 * Decode SentencePiece token IDs to text (vocab from JSON array of subword strings).
 */
export class SentencePieceDecoder {
  private readonly vocab: string[];

  constructor(vocabJson: string) {
    this.vocab = JSON.parse(vocabJson) as string[];
  }

  decode(tokenIds: number[]): string {
    return tokenIds
      .map(id => this.vocab[id] ?? '')
      .join('')
      .replace(/▁/g, ' ')
      .trim();
  }
}