#!/usr/bin/env python3
"""
Dump intermediate encoder activations for parity debugging (golden references).

Requires NeMo + a checkpoint. Saves NumPy .npz files per layer boundary.

Usage:
  python export_golden_references.py --model nvidia/parakeet_realtime_eou_120m-v1 --wav sample.wav --out goldens/
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--wav", required=True)
    p.add_argument("--out", default="goldens")
    args = p.parse_args()

    import numpy as np
    import soundfile as sf
    import torch
    import nemo.collections.asr as nemo_asr

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = nemo_asr.models.ASRModel.from_pretrained(args.model)
    model.eval()
    audio, sr = sf.read(args.wav)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != model.cfg.sample_rate:
        raise SystemExit(f"Resample WAV to {model.cfg.sample_rate} Hz first")

    with torch.no_grad():
        sig = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
        lens = torch.tensor([sig.shape[1]], dtype=torch.int64)
        processed, plen = model.preprocessor(input_signal=sig, length=lens)
        np.savez(out_dir / "preprocessor.npz", features=processed.cpu().numpy())

        encoded, elen = model.encoder(audio_signal=processed, length=plen)
        np.savez(out_dir / "encoder_out.npz", encoded=encoded.cpu().numpy())

    print(f"Wrote preprocessor + encoder tensors under {out_dir}")


if __name__ == "__main__":
    main()
