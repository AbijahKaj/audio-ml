#!/usr/bin/env python3
"""
Download a NeMo model from HuggingFace and export to SafeTensors + config + vocab.
Works without NeMo installed — directly extracts from the .nemo archive.
"""

import json
import os
import sys
import tarfile
import tempfile
import yaml
import struct

import torch
import sentencepiece as spm
from safetensors.torch import save_file
from huggingface_hub import hf_hub_download


def download_nemo(model_id: str, cache_dir: str = "/tmp/nemo_cache") -> str:
    """Download .nemo file from HuggingFace."""
    print(f"Downloading model: {model_id}")
    
    # Try to find the .nemo file
    from huggingface_hub import list_repo_files
    files = list_repo_files(model_id)
    nemo_files = [f for f in files if f.endswith(".nemo")]
    
    if not nemo_files:
        raise RuntimeError(f"No .nemo file found in {model_id}. Files: {files}")
    
    nemo_file = nemo_files[0]
    print(f"Found: {nemo_file}")
    
    path = hf_hub_download(model_id, nemo_file, cache_dir=cache_dir)
    print(f"Downloaded to: {path}")
    return path


def extract_nemo(nemo_path: str, extract_dir: str) -> dict:
    """Extract .nemo archive and return paths to components."""
    print(f"Extracting {nemo_path}...")
    os.makedirs(extract_dir, exist_ok=True)
    
    with tarfile.open(nemo_path, "r:*") as tar:
        tar.extractall(extract_dir)
    
    # Find the key files
    result = {}
    for root, dirs, files in os.walk(extract_dir):
        for f in files:
            full_path = os.path.join(root, f)
            if f.endswith(".ckpt") or f == "model_weights.ckpt":
                result["weights"] = full_path
            elif f == "model_config.yaml":
                result["config"] = full_path
            elif f.endswith(".model") and "tokenizer" in full_path.lower():
                result["tokenizer"] = full_path
            elif f.endswith(".model") and "spe" in f.lower():
                result["tokenizer"] = full_path
            elif f.endswith(".model"):
                # Could be a sentencepiece model
                if "tokenizer" not in result:
                    result["tokenizer"] = full_path
    
    print(f"Found components: {list(result.keys())}")
    for k, v in result.items():
        print(f"  {k}: {v}")
    
    return result


def load_config(config_path: str) -> dict:
    """Load and parse model_config.yaml."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def detect_decoder_type(cfg: dict) -> str:
    """Detect RNNT vs TDT from config."""
    # Check for TDT indicators
    cfg_str = str(cfg).lower()
    if "tdt" in cfg_str:
        return "tdt"
    
    # Check model_defaults for tdt_durations
    model_defaults = cfg.get("model_defaults", {})
    if model_defaults and "tdt_durations" in str(model_defaults):
        return "tdt"
    
    return "rnnt"


def export_config(cfg: dict, decoder_type: str, output_path: str):
    """Export model config as JSON."""
    encoder_cfg = cfg.get("encoder", {})
    preprocessor_cfg = cfg.get("preprocessor", {})
    decoder_cfg = cfg.get("decoder", {})
    joint_cfg = cfg.get("joint", {})
    model_defaults = cfg.get("model_defaults", {})
    
    # Get prediction network config
    pred_cfg = decoder_cfg.get("prednet", {})
    
    config = {
        "encoder_layers": encoder_cfg.get("n_layers", 17),
        "d_model": encoder_cfg.get("d_model", 512),
        "num_heads": encoder_cfg.get("n_heads", 8),
        "conv_kernel_size": encoder_cfg.get("conv_kernel_size", 9),
        "ff_expansion_factor": encoder_cfg.get("ff_expansion_factor", 4),
        "subsampling_factor": encoder_cfg.get("subsampling_factor", 8),
        "vocab_size": joint_cfg.get("vocabulary_size", 
                      decoder_cfg.get("vocabulary_size",
                      model_defaults.get("vocabulary_size", 1025))),
        "pred_hidden": pred_cfg.get("pred_hidden", 640),
        "pred_num_layers": pred_cfg.get("pred_num_layers", 1),
        "joint_dim": joint_cfg.get("joint_hidden", 640),
        "num_mel_bands": preprocessor_cfg.get("features", 80),
        "sample_rate": preprocessor_cfg.get("sample_rate", 16000),
        "window_size_ms": preprocessor_cfg.get("window_size", 0.025) * 1000
            if preprocessor_cfg.get("window_size", 0.025) < 1
            else preprocessor_cfg.get("window_size", 25),
        "hop_size_ms": preprocessor_cfg.get("hop_length", 0.01) * 1000
            if preprocessor_cfg.get("hop_length", 0.01) < 1
            else preprocessor_cfg.get("hop_length", 10),
        "att_context_size": encoder_cfg.get("att_context_size", [70, 1]),
        "decoder_type": decoder_type,
        "subsampling_conv_channels": encoder_cfg.get("subsampling_conv_channels", 256),
    }
    
    if decoder_type == "tdt":
        durations = model_defaults.get("tdt_durations", [0, 1, 2, 3, 4])
        config["tdt_num_durations"] = list(durations) if not isinstance(durations, list) else durations
    
    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"Exported config to {output_path}")
    print(f"  Encoder layers: {config['encoder_layers']}")
    print(f"  d_model: {config['d_model']}")
    print(f"  Decoder type: {config['decoder_type']}")
    print(f"  Vocab size: {config['vocab_size']}")
    
    return config


def export_weights(weights_path: str, output_path: str):
    """Load PyTorch checkpoint and save as SafeTensors."""
    print(f"Loading weights from {weights_path}...")
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
    
    # Some NeMo checkpoints wrap in a "state_dict" key
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    
    # Make all tensors contiguous and float32
    clean_state = {}
    for key, tensor in state_dict.items():
        if isinstance(tensor, torch.Tensor):
            clean_state[key] = tensor.contiguous().float()
    
    print(f"  Total keys: {len(clean_state)}")
    total_params = sum(t.numel() for t in clean_state.values())
    print(f"  Total parameters: {total_params:,}")
    
    # Print all keys with shapes for debugging
    keys_path = output_path.replace(".safetensors", "_keys.txt")
    with open(keys_path, "w") as f:
        for key in sorted(clean_state.keys()):
            shape = list(clean_state[key].shape)
            f.write(f"{key}: {shape}\n")
    print(f"  Saved key listing to {keys_path}")
    
    save_file(clean_state, output_path)
    file_size = os.path.getsize(output_path)
    print(f"  Saved SafeTensors to {output_path} ({file_size / 1024 / 1024:.1f} MB)")


def export_vocab(tokenizer_path: str, output_path: str):
    """Export SentencePiece vocabulary as JSON."""
    print(f"Loading tokenizer from {tokenizer_path}...")
    sp = spm.SentencePieceProcessor()
    sp.Load(tokenizer_path)
    
    vocab = []
    for i in range(sp.GetPieceSize()):
        vocab.append(sp.IdToPiece(i))
    
    # Ensure blank token is at index 0
    if len(vocab) > 0 and vocab[0] != "<blank>":
        vocab = ["<blank>"] + vocab
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)
    
    print(f"Exported {len(vocab)} tokens to {output_path}")
    print(f"  First 10: {vocab[:10]}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HuggingFace model ID")
    parser.add_argument("--output-dir", default="/workspace/test_model", help="Output directory")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    extract_dir = os.path.join(args.output_dir, "extracted")
    
    # Download
    nemo_path = download_nemo(args.model)
    
    # Extract
    components = extract_nemo(nemo_path, extract_dir)
    
    if "config" not in components:
        print("ERROR: No model_config.yaml found in archive!")
        # List what we found
        for root, dirs, files in os.walk(extract_dir):
            for f in files:
                print(f"  {os.path.join(root, f)}")
        sys.exit(1)
    
    # Load config
    cfg = load_config(components["config"])
    decoder_type = detect_decoder_type(cfg)
    
    # Export config
    config = export_config(cfg, decoder_type, os.path.join(args.output_dir, "model_config.json"))
    
    # Export weights
    if "weights" in components:
        export_weights(components["weights"], os.path.join(args.output_dir, "model.safetensors"))
    else:
        print("WARNING: No weights checkpoint found!")
    
    # Export vocab
    if "tokenizer" in components:
        export_vocab(components["tokenizer"], os.path.join(args.output_dir, "vocab.json"))
    else:
        print("WARNING: No tokenizer model found!")
    
    print(f"\n=== Export complete ===")
    print(f"Output directory: {args.output_dir}")
    for f in os.listdir(args.output_dir):
        if not os.path.isdir(os.path.join(args.output_dir, f)):
            size = os.path.getsize(os.path.join(args.output_dir, f))
            print(f"  {f}: {size / 1024 / 1024:.1f} MB" if size > 1024*1024 else f"  {f}: {size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
