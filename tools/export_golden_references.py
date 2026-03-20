#!/usr/bin/env python3
"""
Placeholder: dump intermediate activations from NeMo for layer-by-layer parity checks.

Wire this to your model + test WAV; see plans/fastconformer-asr.md Phase 0.
"""
from __future__ import annotations

import argparse


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--wav", required=True)
    p.add_argument("--out", default="golden.npz")
    args = p.parse_args()
    print("TODO: implement golden export using NeMo forward hooks.", args.wav, "->", args.out)


if __name__ == "__main__":
    main()
