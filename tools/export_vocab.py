"""
Export the SentencePiece vocabulary from a NeMo model as a JSON array.
The output is a simple array where index = token ID, value = token string.
Token 0 is always the blank token.

Usage:
    pip install nemo_toolkit[asr]
    python export_vocab.py --model nvidia/parakeet_realtime_eou_120m-v1 --out vocab.json
"""

import argparse
import json
import nemo.collections.asr as nemo_asr


def export_vocab(model_name: str, out_path: str):
    print(f"Loading model: {model_name}")
    model = nemo_asr.models.ASRModel.from_pretrained(model_name)

    # Try different ways to get the vocabulary
    vocab = None

    # Method 1: decoding.vocabulary (common for CTC/RNNT models)
    if hasattr(model, "decoder") and hasattr(model.decoder, "vocabulary"):
        vocab = model.decoder.vocabulary
        print(f"  Found vocabulary on model.decoder.vocabulary ({len(vocab)} tokens)")

    # Method 2: tokenizer
    elif hasattr(model, "tokenizer") and hasattr(model.tokenizer, "vocab"):
        vocab_dict = model.tokenizer.vocab
        vocab = [None] * (max(vocab_dict.values()) + 2)
        vocab[0] = "<blank>"
        for token, idx in vocab_dict.items():
            vocab[idx + 1] = token
        print(f"  Found vocabulary on model.tokenizer.vocab ({len(vocab)} tokens)")

    # Method 3: from tokenizer ids_to_tokens
    elif hasattr(model, "tokenizer"):
        tokenizer = model.tokenizer
        size = tokenizer.vocab_size
        vocab = ["<blank>"]
        for i in range(size):
            try:
                vocab.append(tokenizer.ids_to_tokens([i])[0])
            except Exception:
                vocab.append(f"<unk_{i}>")
        print(f"  Found vocabulary via tokenizer.ids_to_tokens ({len(vocab)} tokens)")

    else:
        raise ValueError("Cannot find vocabulary on this model. Try inspecting model.decoder.")

    # Clean up None entries
    vocab = [v if v is not None else f"<null_{i}>" for i, v in enumerate(vocab)]

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    print(f"  Vocabulary saved to {out_path}")
    print(f"  Sample tokens: {vocab[:10]}")
    print(f"  Total tokens: {len(vocab)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="nvidia/parakeet_realtime_eou_120m-v1")
    parser.add_argument("--out", default="vocab.json")
    args = parser.parse_args()
    export_vocab(args.model, args.out)
