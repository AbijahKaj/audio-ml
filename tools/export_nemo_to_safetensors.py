#!/usr/bin/env python3
"""
Export NeMo ASR model checkpoints to SafeTensors format + model config JSON.

Usage:
    python export_nemo_to_safetensors.py --model nvidia/parakeet_realtime_eou_120m-v1
    python export_nemo_to_safetensors.py --model nvidia/stt_en_fastconformer_tdt_large
    python export_nemo_to_safetensors.py --model nvidia/parakeet-tdt-0.6b-v3

Requirements:
    pip install nemo_toolkit[asr] safetensors torch
"""

import argparse
import json
import sys

import torch
import nemo.collections.asr as nemo_asr
from safetensors.torch import save_file


def detect_decoder_type(model):
    """Detect whether the model uses RNNT or TDT decoder."""
    if hasattr(model.cfg, "tdt"):
        return "tdt"
    if hasattr(model, "joint") and hasattr(model.joint, "duration_head"):
        return "tdt"
    cfg_str = str(model.cfg)
    if "tdt" in cfg_str.lower():
        return "tdt"
    return "rnnt"


def extract_config(model, decoder_type):
    """Extract model configuration as a JSON-serializable dict."""
    cfg = model.cfg

    config = {
        "encoder_layers": cfg.encoder.n_layers,
        "d_model": cfg.encoder.d_model,
        "num_heads": cfg.encoder.n_heads,
        "conv_kernel_size": cfg.encoder.conv_kernel_size,
        "ff_expansion_factor": cfg.encoder.ff_expansion_factor,
        "subsampling_factor": getattr(cfg.encoder, "subsampling_factor", 8),
        "vocab_size": model.decoder.prediction.embed.weight.shape[0],
        "pred_hidden": model.decoder.prediction.lstm.hidden_size
            if hasattr(model.decoder.prediction, "lstm") else cfg.decoder.prednet.pred_hidden,
        "pred_num_layers": getattr(cfg.decoder.prednet, "pred_num_layers", 1),
        "joint_dim": getattr(cfg.joint, "joint_hidden", 640),
        "num_mel_bands": cfg.preprocessor.features,
        "sample_rate": cfg.preprocessor.sample_rate,
        "window_size_ms": cfg.preprocessor.window_size * 1000
            if cfg.preprocessor.window_size < 1 else cfg.preprocessor.window_size,
        "hop_size_ms": getattr(cfg.preprocessor, "hop_length_ms", 10),
        "att_context_size": list(cfg.encoder.att_context_size)
            if hasattr(cfg.encoder, "att_context_size") else [70, 1],
        "decoder_type": decoder_type,
        "subsampling_conv_channels": getattr(cfg.encoder, "subsampling_conv_channels", 256),
    }

    if decoder_type == "tdt":
        durations = getattr(cfg, "tdt_durations", None)
        if durations is None:
            durations = getattr(cfg.model_defaults, "tdt_durations", [0, 1, 2, 3, 4])
        config["tdt_num_durations"] = list(durations)

    return config


def main():
    parser = argparse.ArgumentParser(description="Export NeMo model to SafeTensors")
    parser.add_argument("--model", required=True, help="HuggingFace model ID or local .nemo path")
    parser.add_argument("--output-prefix", default=None, help="Output file prefix (default: derived from model name)")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model = nemo_asr.models.ASRModel.from_pretrained(args.model)
    model.eval()

    prefix = args.output_prefix
    if prefix is None:
        prefix = args.model.replace("/", "_").replace("-", "_")

    decoder_type = detect_decoder_type(model)
    print(f"Detected decoder type: {decoder_type}")

    # Export weights
    print("Exporting weights...")
    state_dict = {}
    for key, tensor in model.state_dict().items():
        state_dict[key] = tensor.contiguous().cpu()

    safetensors_path = f"{prefix}.safetensors"
    save_file(state_dict, safetensors_path)
    print(f"Saved weights to {safetensors_path}")

    # Export config
    config = extract_config(model, decoder_type)
    config_path = f"{prefix}_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved config to {config_path}")

    # Print all weight keys for reference
    keys_path = f"{prefix}_keys.txt"
    with open(keys_path, "w") as f:
        for key in sorted(state_dict.keys()):
            shape = list(state_dict[key].shape)
            f.write(f"{key}: {shape}\n")
    print(f"Saved weight keys to {keys_path}")

    print(f"\nModel summary:")
    print(f"  Decoder type: {decoder_type}")
    print(f"  Encoder layers: {config['encoder_layers']}")
    print(f"  d_model: {config['d_model']}")
    print(f"  Vocab size: {config['vocab_size']}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Total weight keys: {len(state_dict)}")


if __name__ == "__main__":
    main()
