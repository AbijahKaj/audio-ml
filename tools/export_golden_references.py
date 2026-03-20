#!/usr/bin/env python3
"""
Export golden intermediate tensors from a NeMo FastConformer model.

This script stores:
- input log-mel features
- pre-encoder output
- each conformer block output
- final encoder output
"""

from __future__ import annotations

import argparse
import json
import os
import wave
from typing import Dict, List

import numpy as np
import torch


def read_wav_mono(path: str) -> tuple[np.ndarray, int]:
    with wave.open(path, "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_rate = wav_file.getframerate()
        sample_width = wav_file.getsampwidth()
        frame_count = wav_file.getnframes()
        raw = wav_file.readframes(frame_count)

    if sample_width != 2:
        raise ValueError(f"Only 16-bit PCM WAV is supported, got sample width: {sample_width}")

    data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return data, sample_rate


def resample_linear(audio: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    if from_rate == to_rate:
        return audio
    ratio = to_rate / from_rate
    out_len = int(round(len(audio) * ratio))
    positions = np.arange(out_len, dtype=np.float32) / ratio
    left = np.floor(positions).astype(np.int64)
    right = np.clip(left + 1, 0, len(audio) - 1)
    frac = positions - left
    left = np.clip(left, 0, len(audio) - 1)
    return ((1.0 - frac) * audio[left] + frac * audio[right]).astype(np.float32)


def ensure_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Export NeMo golden references")
    parser.add_argument("--model", required=True, help="NeMo pretrained model name")
    parser.add_argument("--wav", required=True, help="Input WAV file")
    parser.add_argument("--out-dir", required=True, help="Directory where tensors are saved")
    args = parser.parse_args()

    try:
        import nemo.collections.asr as nemo_asr
    except Exception as exc:
        raise RuntimeError("nemo_toolkit[asr] is required for this script") from exc

    os.makedirs(args.out_dir, exist_ok=True)

    model = nemo_asr.models.ASRModel.from_pretrained(args.model)
    model.eval()
    target_sr = int(model.cfg.preprocessor.sample_rate)

    audio, source_sr = read_wav_mono(args.wav)
    audio = resample_linear(audio, source_sr, target_sr)

    signal = torch.from_numpy(audio).unsqueeze(0)
    lengths = torch.tensor([signal.shape[1]], dtype=torch.int64)

    layer_outputs: Dict[int, np.ndarray] = {}
    hooks: List[torch.utils.hooks.RemovableHandle] = []

    def _make_hook(index: int):
        def _hook(_module, _inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            layer_outputs[index] = ensure_numpy(tensor)
        return _hook

    for idx, layer in enumerate(model.encoder.layers):
        hooks.append(layer.register_forward_hook(_make_hook(idx)))

    with torch.no_grad():
        features, feature_lengths = model.preprocessor(input_signal=signal, length=lengths)
        pre_encoded = model.encoder.pre_encode(features)
        encoded, encoded_len = model.encoder(audio_signal=features, length=feature_lengths)

    for hook in hooks:
        hook.remove()

    np.save(os.path.join(args.out_dir, "audio.npy"), audio)
    np.save(os.path.join(args.out_dir, "features.npy"), ensure_numpy(features))
    np.save(os.path.join(args.out_dir, "pre_encode.npy"), ensure_numpy(pre_encoded))
    np.save(os.path.join(args.out_dir, "encoder_output.npy"), ensure_numpy(encoded))
    np.save(os.path.join(args.out_dir, "encoder_output_len.npy"), ensure_numpy(encoded_len))

    layer_dir = os.path.join(args.out_dir, "layers")
    os.makedirs(layer_dir, exist_ok=True)
    for idx, tensor in sorted(layer_outputs.items()):
        np.save(os.path.join(layer_dir, f"layer_{idx:02d}.npy"), tensor)

    metadata = {
        "model": args.model,
        "input_wav": args.wav,
        "sample_rate": target_sr,
        "num_layers": len(layer_outputs),
        "feature_shape": list(features.shape),
        "encoder_shape": list(encoded.shape),
    }
    with open(os.path.join(args.out_dir, "metadata.json"), "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)

    print(f"Saved golden references to {args.out_dir}")


if __name__ == "__main__":
    main()
