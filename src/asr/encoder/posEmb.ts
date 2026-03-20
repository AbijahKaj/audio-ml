/** NeMo-style sinusoidal table for LocalAttRelPositionalEncoding (fixed length). */
export function createLocalRelativePositionalEncoding(
  leftContext: number,
  rightContext: number,
  dModel: number,
): Float32Array {
  const rows = leftContext + rightContext + 1;
  const pe = new Float32Array(rows * dModel);
  const log10000 = Math.log(10000);
  let row = 0;
  for (let pos = leftContext; pos >= -rightContext - 1; pos--) {
    for (let j = 0; j < dModel; j += 2) {
      const div = Math.exp((j * -log10000) / dModel);
      pe[row * dModel + j] = Math.sin(pos * div);
      if (j + 1 < dModel) {
        pe[row * dModel + j + 1] = Math.cos(pos * div);
      }
    }
    row++;
  }
  return pe;
}

/** Relative strip for RelPositionalEncoding offline (2*maxLen - 1 positions). */
export function createExtendedRelativePositionalEncoding(maxLen: number, dModel: number): Float32Array {
  const n = 2 * maxLen - 1;
  const pe = new Float32Array(n * dModel);
  const log10000 = Math.log(10000);
  for (let i = 0; i < n; i++) {
    const pos = maxLen - 1 - i;
    for (let j = 0; j < dModel; j += 2) {
      const div = Math.exp((j * -log10000) / dModel);
      pe[i * dModel + j] = Math.sin(pos * div);
      if (j + 1 < dModel) {
        pe[i * dModel + j + 1] = Math.cos(pos * div);
      }
    }
  }
  return pe;
}
