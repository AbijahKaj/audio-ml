#!/usr/bin/env python3
"""
Export SentencePiece vocabulary from a NeMo ASR model.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, List


def resolve_tokenizer(model: Any) -> Any:
    if hasattr(model, "tokenizer"):
        return model.tokenizer
    decoder = getattr(model, "decoder", None)
    if decoder is not None and hasattr(decoder, "tokenizer"):
        return decoder.tokenizer
    raise RuntimeError("Could not locate tokenizer in the provided model.")


def export_vocab_list(tokenizer: Any) -> List[str]:
    if hasattr(tokenizer, "vocab"):
        vocab = tokenizer.vocab
        if isinstance(vocab, list):
            return vocab
        if isinstance(vocab, dict):
            size = max(vocab.values()) + 1
            out = [""] * size
            for token, index in vocab.items():
                out[index] = token
            return out

    inner = getattr(tokenizer, "tokenizer", None)
    if inner is not None:
        if hasattr(inner, "get_piece_size") and hasattr(inner, "id_to_piece"):
            size = inner.get_piece_size()
            return [inner.id_to_piece(i) for i in range(size)]
        if hasattr(inner, "piece_to_id") and hasattr(inner, "vocab_size"):
            size = int(inner.vocab_size())
            return [inner.id_to_piece(i) for i in range(size)]

    if hasattr(tokenizer, "ids_to_tokens") and hasattr(tokenizer, "vocab_size"):
        size = int(tokenizer.vocab_size)
        return [tokenizer.ids_to_tokens([i])[0] for i in range(size)]

    raise RuntimeError("Unsupported tokenizer type; cannot export vocabulary.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export NeMo tokenizer vocabulary")
    parser.add_argument("--model", required=True, help="NeMo pretrained model name")
    parser.add_argument("--out", required=True, help="Output JSON path")
    args = parser.parse_args()

    try:
        import nemo.collections.asr as nemo_asr
    except Exception as exc:
        raise RuntimeError("nemo_toolkit[asr] is required for this script") from exc

    model = nemo_asr.models.ASRModel.from_pretrained(args.model)
    tokenizer = resolve_tokenizer(model)
    vocab = export_vocab_list(tokenizer)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as file:
        json.dump(vocab, file, indent=2, ensure_ascii=False)

    print(f"Exported {len(vocab)} vocab entries to {args.out}")


if __name__ == "__main__":
    main()
