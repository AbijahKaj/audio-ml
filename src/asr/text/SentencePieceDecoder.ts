const SPACE_MARKER = '▁';

function isSpecialToken(piece: string): boolean {
  return piece.startsWith('<') && piece.endsWith('>');
}

/**
 * SentencePiece decode-side helper.
 */
export class SentencePieceDecoder {
  private vocab: string[];

  constructor(vocabSource: string | string[]) {
    this.vocab = Array.isArray(vocabSource) ? vocabSource : (JSON.parse(vocabSource) as string[]);
  }

  decode(tokenIds: number[]): string {
    return tokenIds
      .map((id) => this.vocab[id] ?? '')
      .filter((piece) => piece.length > 0 && !isSpecialToken(piece))
      .join('')
      .replaceAll(SPACE_MARKER, ' ')
      .replace(/\s+/g, ' ')
      .trim();
  }
}
