#!/usr/bin/env python3
"""
Convert a NeMo FastConformer ASR model to the format expected by audio-ml:
  - model.safetensors   (all weights, flat NeMo key names)
  - model_config.json   (architecture hyperparameters)
  - vocab.json          (token_id → text mapping)

Usage:
  # Requires: pip install nemo_toolkit[asr] safetensors sentencepiece

  # Export the small 110M TDT model (browser-friendly)
  python tools/export_nemo_to_safetensors.py \
      --model nvidia/parakeet-tdt_ctc-110m \
      --output-dir exported/parakeet-tdt-110m

  # Export the 120M RNNT streaming model
  python tools/export_nemo_to_safetensors.py \
      --model nvidia/parakeet_realtime_eou_120m-v1 \
      --output-dir exported/parakeet-rnnt-120m

  # Then upload to HuggingFace:
  huggingface-cli upload YOUR_USER/parakeet-tdt-110m-safetensors exported/parakeet-tdt-110m .
"""
import argparse
import json
import os
import sys


def detect_decoder_type(model) -> str:
    cfg = model.cfg
    if hasattr(cfg, "tdt") or hasattr(cfg, "model_defaults") and hasattr(cfg.model_defaults, "tdt_durations"):
        return "tdt"
    if hasattr(model, "joint") and hasattr(model.joint, "duration_head"):
        return "tdt"
    joint_cfg = getattr(cfg, "joint", None)
    if joint_cfg and hasattr(joint_cfg, "num_extra_outputs") and joint_cfg.num_extra_outputs > 0:
        return "tdt"
    return "rnnt"


def get_tdt_durations(model) -> list[int] | None:
    cfg = model.cfg
    defaults = getattr(cfg, "model_defaults", None)
    if defaults and hasattr(defaults, "tdt_durations"):
        return list(defaults.tdt_durations)
    if hasattr(cfg, "tdt") and hasattr(cfg.tdt, "durations"):
        return list(cfg.tdt.durations)
    return None


def extract_config(model) -> dict:
    cfg = model.cfg
    enc = cfg.encoder

    decoder_type = detect_decoder_type(model)

    pred = cfg.decoder.get("prednet", cfg.decoder) if hasattr(cfg.decoder, "prednet") else cfg.decoder
    pred_hidden = getattr(pred, "pred_hidden", None) or getattr(pred, "hidden_size", 640)
    pred_num_layers = getattr(pred, "pred_rnn_layers", None) or getattr(pred, "num_layers", 1)

    joint_dim = getattr(cfg.joint, "joint_hidden", None) or getattr(cfg.joint, "jointnet", {}).get("joint_hidden", 640)
    if hasattr(joint_dim, "__int__"):
        joint_dim = int(joint_dim)
    else:
        joint_dim = 640

    vocab_size = model.tokenizer.vocab_size
    blank_appended = True
    if hasattr(model.decoder, "blank_idx"):
        blank_appended = model.decoder.blank_idx == vocab_size

    result = {
        "encoder_layers": int(enc.n_layers),
        "d_model": int(enc.d_model),
        "num_heads": int(enc.n_heads),
        "conv_kernel_size": int(enc.conv_kernel_size),
        "ff_expansion_factor": int(getattr(enc, "ff_expansion_factor", 4)),
        "subsampling_factor": int(getattr(enc, "subsampling_factor", 8)),
        "subsampling_conv_channels": int(getattr(enc, "subsampling_conv_channels", 256)),
        "vocab_size": vocab_size + (1 if blank_appended else 0),
        "pred_hidden": int(pred_hidden),
        "pred_num_layers": int(pred_num_layers),
        "joint_dim": int(joint_dim),
        "num_mel_bands": int(cfg.preprocessor.features),
        "sample_rate": int(cfg.preprocessor.sample_rate),
        "window_size_ms": float(cfg.preprocessor.window_size) * 1000
            if cfg.preprocessor.window_size < 1
            else float(cfg.preprocessor.window_size),
        "hop_size_ms": float(getattr(cfg.preprocessor, "hop_length_ms", 10)),
        "att_context_size": list(getattr(enc, "att_context_size", [70, 1])),
        "decoder_type": decoder_type,
    }

    if decoder_type == "tdt":
        durations = get_tdt_durations(model)
        if durations:
            result["tdt_num_durations"] = durations

    # Feature normalization mode (streaming models typically use normalize: "NA")
    normalize = getattr(cfg.preprocessor, "normalize", "per_feature")
    result["normalize"] = str(normalize) if normalize else "per_feature"

    return result


def extract_vocab(model) -> dict[str, str]:
    """Build {token_id: text} map from the tokenizer."""
    tokenizer = model.tokenizer
    vocab = {}
    vocab_size = tokenizer.vocab_size

    for i in range(vocab_size):
        try:
            piece = tokenizer.ids_to_tokens([i])
            if isinstance(piece, list):
                piece = piece[0] if piece else f"<unk_{i}>"
            vocab[str(i)] = str(piece)
        except Exception:
            vocab[str(i)] = f"<unk_{i}>"

    blank_id = vocab_size
    vocab[str(blank_id)] = "<blank>"

    return vocab


def export_weights(model, output_path: str):
    from safetensors.torch import save_file
    state_dict = {}
    for k, v in model.state_dict().items():
        state_dict[k] = v.contiguous().half()
    save_file(state_dict, output_path)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Weights: {output_path} ({size_mb:.1f} MB, {len(state_dict)} tensors, float16)")


def main():
    parser = argparse.ArgumentParser(description="Export NeMo FastConformer to SafeTensors + config + vocab")
    parser.add_argument("--model", required=True,
                        help="HuggingFace model ID (e.g. nvidia/parakeet-tdt_ctc-110m) or local .nemo path")
    parser.add_argument("--output-dir", required=True, help="Output directory for exported files")
    parser.add_argument("--fp32", action="store_true", help="Export weights in float32 (default: float16)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model: {args.model}")
    import nemo.collections.asr as nemo_asr
    if args.model.endswith(".nemo"):
        model = nemo_asr.models.ASRModel.restore_from(args.model)
    else:
        model = nemo_asr.models.ASRModel.from_pretrained(args.model)
    model.eval()
    print(f"  Model loaded: {type(model).__name__}")

    config = extract_config(model)
    config_path = os.path.join(args.output_dir, "model_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Config: {config_path}")
    print(f"    encoder: {config['encoder_layers']} layers, d_model={config['d_model']}")
    print(f"    decoder: {config['decoder_type']}, vocab={config['vocab_size']}, pred_hidden={config['pred_hidden']}")

    vocab = extract_vocab(model)
    vocab_path = os.path.join(args.output_dir, "vocab.json")
    with open(vocab_path, "w") as f:
        json.dump(vocab, f, ensure_ascii=False)
    print(f"  Vocab: {vocab_path} ({len(vocab)} tokens)")

    weights_path = os.path.join(args.output_dir, "model.safetensors")
    if args.fp32:
        from safetensors.torch import save_file
        state_dict = {k: v.contiguous() for k, v in model.state_dict().items()}
        save_file(state_dict, weights_path)
        size_mb = os.path.getsize(weights_path) / (1024 * 1024)
        print(f"  Weights: {weights_path} ({size_mb:.1f} MB, {len(state_dict)} tensors, float32)")
    else:
        export_weights(model, weights_path)

    print(f"\nDone! Upload to HuggingFace with:")
    print(f"  huggingface-cli upload YOUR_USER/MODEL_NAME {args.output_dir} .")
    print(f"\nThen set the demo model URLs to:")
    print(f"  https://huggingface.co/YOUR_USER/MODEL_NAME/resolve/main/model_config.json")
    print(f"  https://huggingface.co/YOUR_USER/MODEL_NAME/resolve/main/model.safetensors")
    print(f"  https://huggingface.co/YOUR_USER/MODEL_NAME/resolve/main/vocab.json")


if __name__ == "__main__":
    main()
