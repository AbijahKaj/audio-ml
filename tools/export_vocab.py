#!/usr/bin/env python3
"""Export tokenizer vocabulary to JSON array for SentencePieceDecoder."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", required=True, help="Path to tokenizer.model or vocab file")
    p.add_argument("--out", default="vocab.json", help="Output JSON path")
    args = p.parse_args()

    try:
        import sentencepiece as spm
    except ImportError:
        raise SystemExit("pip install sentencepiece") from None

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer)
    vocab = [sp.id_to_piece(i) for i in range(sp.get_piece_size())]
    Path(args.out).write_text(json.dumps(vocab))
    print(f"Wrote {len(vocab)} entries to {args.out}")


if __name__ == "__main__":
    main()
