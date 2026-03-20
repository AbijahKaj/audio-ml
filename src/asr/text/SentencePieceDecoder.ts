export class SentencePieceDecoder {
  private readonly vocab: string[];

  constructor(vocabJson: string) {
    const raw = JSON.parse(vocabJson) as unknown;
    if (Array.isArray(raw)) {
      this.vocab = raw.map(String);
    } else if (raw && typeof raw === 'object') {
      const entries = Object.entries(raw as Record<string, string>);
      entries.sort(([a], [b]) => Number(a) - Number(b));
      this.vocab = entries.map(([, v]) => String(v));
    } else {
      throw new Error('vocab.json must be an array of strings or id→token map');
    }
  }

  decode(tokenIds: number[]): string {
    return tokenIds
      .map(id => this.vocab[id] ?? '')
      .join('')
      .replace(/▁/g, ' ')
      .trim();
  }
}
