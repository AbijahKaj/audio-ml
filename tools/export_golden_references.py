#!/usr/bin/env python3
"""Export reference tensors from NeMo preprocessing and encoder passes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nemo.collections.asr as nemo_asr
import soundfile as sf
import torch


def tensor_to_list(tensor: torch.Tensor) -> list[float] | list[list[float]] | list[list[list[float]]]:
    return tensor.detach().cpu().tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="NeMo/Hugging Face model name")
    parser.add_argument("--wav", required=True, help="Wave file to trace")
    parser.add_argument("--output", default="golden_references.json", help="Destination JSON path")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = nemo_asr.models.ASRModel.from_pretrained(args.model)
    audio, sample_rate = sf.read(args.wav, dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]

    audio_tensor = torch.from_numpy(audio).unsqueeze(0)
    length_tensor = torch.tensor([audio_tensor.shape[1]], dtype=torch.int64)

    with torch.no_grad():
        processed, processed_len = model.preprocessor(
            input_signal=audio_tensor,
            length=length_tensor,
        )
        encoded, encoded_len = model.encoder(audio_signal=processed, length=processed_len)

    payload = {
        "sample_rate": int(sample_rate),
        "preprocessor_output": tensor_to_list(processed),
        "preprocessor_length": tensor_to_list(processed_len),
        "encoder_output": tensor_to_list(encoded),
        "encoder_length": tensor_to_list(encoded_len),
    }

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle)


if __name__ == "__main__":
    main()
