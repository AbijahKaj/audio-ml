#!/usr/bin/env python3
"""Export a NeMo tokenizer vocabulary into JSON for the JS decoder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nemo.collections.asr as nemo_asr


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="NeMo/Hugging Face model name")
    parser.add_argument("--output", default="vocab.json", help="Destination JSON path")
    args = parser.parse_args()

    model = nemo_asr.models.ASRModel.from_pretrained(args.model)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if hasattr(model.tokenizer, "tokenizer") and hasattr(model.tokenizer.tokenizer, "id_to_piece"):
        vocab = [model.tokenizer.tokenizer.id_to_piece(index) for index in range(model.tokenizer.vocab_size)]
    else:
        vocab = [token for token, _ in sorted(model.tokenizer.tokenizer.get_vocab().items(), key=lambda entry: entry[1])]

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(vocab, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
