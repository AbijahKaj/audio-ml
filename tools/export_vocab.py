#!/usr/bin/env python3
"""
Export SentencePiece vocabulary from a NeMo model as a JSON lookup table.

Usage:
    python export_vocab.py --model nvidia/parakeet_realtime_eou_120m-v1
    python export_vocab.py --model nvidia/parakeet-tdt-0.6b-v3

Requirements:
    pip install nemo_toolkit[asr] sentencepiece
"""

import argparse
import json

import nemo.collections.asr as nemo_asr


def main():
    parser = argparse.ArgumentParser(description="Export SentencePiece vocabulary")
    parser.add_argument("--model", required=True, help="HuggingFace model ID or local .nemo path")
    parser.add_argument("--output", default=None, help="Output JSON file (default: derived from model name)")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model = nemo_asr.models.ASRModel.from_pretrained(args.model)

    output = args.output
    if output is None:
        output = args.model.replace("/", "_").replace("-", "_") + "_vocab.json"

    # Extract vocabulary from the tokenizer
    if hasattr(model, "tokenizer"):
        tokenizer = model.tokenizer
    elif hasattr(model, "decoder") and hasattr(model.decoder, "vocabulary"):
        tokenizer = None
        vocab = list(model.decoder.vocabulary)
    else:
        raise RuntimeError("Cannot find tokenizer or vocabulary in model")

    if tokenizer is not None:
        if hasattr(tokenizer, "tokenizer"):
            sp = tokenizer.tokenizer
            vocab = []
            for i in range(sp.get_piece_size()):
                vocab.append(sp.id_to_piece(i))
        elif hasattr(tokenizer, "vocab"):
            vocab = list(tokenizer.vocab)
        else:
            vocab = [tokenizer.ids_to_tokens([i])[0] for i in range(tokenizer.vocab_size)]

    # Ensure blank token is at index 0
    if len(vocab) > 0 and vocab[0] != "<blank>":
        vocab = ["<blank>"] + vocab

    with open(output, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    print(f"Exported {len(vocab)} tokens to {output}")
    print(f"First 10 tokens: {vocab[:10]}")
    print(f"Last 10 tokens: {vocab[-10:]}")


if __name__ == "__main__":
    main()
