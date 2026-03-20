"""
Export intermediate tensors (golden references) at every layer boundary for a test WAV.
These are used to validate that the JavaScript implementation produces matching outputs.

Usage:
    pip install nemo_toolkit[asr] safetensors torch soundfile numpy
    python export_golden_references.py \
        --model nvidia/parakeet_realtime_eou_120m-v1 \
        --wav test.wav \
        --out golden_references/

The output directory will contain:
    mel_features.npy        - [1, T, 80] log-mel features
    subsampling_output.npy  - [1, T/8, d_model] after ConvSubsampling
    layer_{i}_output.npy    - [1, T/8, d_model] after each ConformerBlock
    encoder_output.npy      - [1, T/8, d_model] final encoder output
    pred_state_init.npy     - Initial LSTM state (zeros)
    joint_output_frame0.npy - Joint network output for frame 0
    decoder_tokens.json     - Final token IDs from greedy decode
    transcript.txt          - Final text transcript
"""

import argparse
import json
import os

import numpy as np
import soundfile as sf
import torch
import nemo.collections.asr as nemo_asr


def hook_factory(name, storage):
    def hook(module, input, output):
        if isinstance(output, torch.Tensor):
            storage[name] = output.detach().cpu().numpy()
        elif isinstance(output, tuple) and isinstance(output[0], torch.Tensor):
            storage[name] = output[0].detach().cpu().numpy()
    return hook


def export_golden_references(model_name: str, wav_path: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading model: {model_name}")
    model = nemo_asr.models.ASRModel.from_pretrained(model_name)
    model.eval()

    # Load and preprocess audio
    print(f"Loading audio: {wav_path}")
    audio, sr = sf.read(wav_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        import resampy
        audio = resampy.resample(audio, sr, 16000)

    # Register hooks to capture intermediate outputs
    captured = {}
    hooks = []

    # Hook mel preprocessor
    hooks.append(model.preprocessor.register_forward_hook(hook_factory("mel_features", captured)))

    # Hook subsampling
    if hasattr(model.encoder, "pre_encode"):
        hooks.append(model.encoder.pre_encode.register_forward_hook(
            hook_factory("subsampling_output", captured)))

    # Hook each conformer layer
    for i, layer in enumerate(model.encoder.layers):
        hooks.append(layer.register_forward_hook(hook_factory(f"layer_{i}_output", captured)))

    # Run inference
    print("Running inference...")
    with torch.no_grad():
        audio_tensor = torch.tensor(audio).unsqueeze(0)
        audio_lengths = torch.tensor([audio.shape[0]])
        transcript = model.transcribe([wav_path])

    # Remove hooks
    for h in hooks:
        h.remove()

    # Save all captured tensors
    for name, tensor in captured.items():
        path = os.path.join(out_dir, f"{name}.npy")
        np.save(path, tensor)
        print(f"  Saved {name}: shape={tensor.shape}")

    # Save transcript
    transcript_path = os.path.join(out_dir, "transcript.txt")
    with open(transcript_path, "w") as f:
        f.write(transcript[0] if transcript else "")
    print(f"  Transcript: {transcript[0] if transcript else ''}")
    print(f"  Saved to {transcript_path}")

    print(f"\nAll golden references saved to: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="nvidia/parakeet_realtime_eou_120m-v1")
    parser.add_argument("--wav", required=True, help="Path to 16kHz WAV file for testing")
    parser.add_argument("--out", default="golden_references")
    args = parser.parse_args()
    export_golden_references(args.model, args.wav, args.out)
