#!/usr/bin/env python3
"""
Export golden reference tensors at every layer boundary for a test WAV file.
Used to validate the JavaScript implementation layer by layer.

Usage:
    python export_golden_references.py --model nvidia/parakeet_realtime_eou_120m-v1 --wav test.wav

Requirements:
    pip install nemo_toolkit[asr] numpy soundfile
"""

import argparse
import json
import os

import numpy as np
import soundfile as sf
import torch
import nemo.collections.asr as nemo_asr


def register_hooks(model, output_dir):
    """Register forward hooks to capture intermediate outputs."""
    saved = {}

    def make_hook(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                tensor = output[0]
            elif isinstance(output, torch.Tensor):
                tensor = output
            else:
                return
            saved[name] = tensor.detach().cpu().numpy()
        return hook

    # Preprocessor output
    if hasattr(model, "preprocessor"):
        model.preprocessor.register_forward_hook(make_hook("preprocessor_output"))

    # Encoder subsampling
    if hasattr(model.encoder, "pre_encode"):
        model.encoder.pre_encode.register_forward_hook(make_hook("encoder.pre_encode"))

    # Each conformer layer
    if hasattr(model.encoder, "layers"):
        for i, layer in enumerate(model.encoder.layers):
            layer.register_forward_hook(make_hook(f"encoder.layers.{i}"))

    # Encoder final output
    model.encoder.register_forward_hook(make_hook("encoder_output"))

    return saved


def main():
    parser = argparse.ArgumentParser(description="Export golden reference tensors")
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument("--wav", required=True, help="Path to test WAV file")
    parser.add_argument("--output-dir", default="golden_refs", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model: {args.model}")
    model = nemo_asr.models.ASRModel.from_pretrained(args.model)
    model.eval()

    saved = register_hooks(model, args.output_dir)

    print(f"Loading audio: {args.wav}")
    audio, sr = sf.read(args.wav, dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]

    print(f"Running inference (sample rate: {sr}, duration: {len(audio)/sr:.2f}s)")
    with torch.no_grad():
        transcription = model.transcribe([args.wav])

    print(f"Transcription: {transcription}")

    # Save all captured tensors
    manifest = {}
    for name, array in saved.items():
        filename = f"{name.replace('.', '_')}.npy"
        filepath = os.path.join(args.output_dir, filename)
        np.save(filepath, array)
        manifest[name] = {
            "file": filename,
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "min": float(array.min()),
            "max": float(array.max()),
            "mean": float(array.mean()),
        }
        print(f"  {name}: shape={array.shape}, range=[{array.min():.6f}, {array.max():.6f}]")

    manifest_path = os.path.join(args.output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Save the raw audio as well
    audio_path = os.path.join(args.output_dir, "input_audio.npy")
    np.save(audio_path, audio)

    # Save transcription
    trans_path = os.path.join(args.output_dir, "transcription.json")
    with open(trans_path, "w") as f:
        json.dump({"text": transcription[0] if isinstance(transcription, list) else str(transcription)}, f, indent=2)

    print(f"\nSaved {len(saved)} reference tensors to {args.output_dir}/")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
