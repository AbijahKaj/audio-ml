export class SentencePieceDecoder {
  private readonly vocab: string[];

  constructor(vocabJson: string) {
    this.vocab = JSON.parse(vocabJson) as string[];
  }

  decode(tokenIds: number[]): string {
    return tokenIds
      .map((tokenId) => this.vocab[tokenId] ?? '')
      .filter((piece) => piece !== '<blank>' && piece !== '<unk>')
      .join('')
      .replace(/▁/g, ' ')
      .replace(/\s+/g, ' ')
      .trim();
  }
}
