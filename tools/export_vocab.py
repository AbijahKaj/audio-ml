#!/usr/bin/env python3
"""
Export SentencePiece vocabulary from a NeMo tokenizer to JSON array for SentencePieceDecoder.

Usage:
  python export_vocab.py --tokenizer /path/to/tokenizer.model --out vocab.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", required=True, help="sentencepiece .model file")
    p.add_argument("--out", required=True, help="Output vocab.json")
    args = p.parse_args()

    import sentencepiece as spm

    sp = spm.SentencePieceProcessor()
    sp.load(args.tokenizer)
    vocab = [sp.id_to_piece(i) for i in range(sp.get_piece_size())]
    Path(args.out).write_text(json.dumps(vocab))
    print(f"Wrote {len(vocab)} entries to {args.out}")


if __name__ == "__main__":
    main()
