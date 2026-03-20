#!/usr/bin/env python3
"""
Export a NeMo ASR checkpoint to SafeTensors + model_config.json.

Usage:
  python tools/export_nemo_to_safetensors.py \
    --model nvidia/parakeet_realtime_eou_120m-v1 \
    --out-dir ./artifacts/parakeet_120m
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict


def _to_plain(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return [_to_plain(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def infer_decoder_type(model: Any) -> str:
    cfg = getattr(model, "cfg", None)
    if cfg is not None and (hasattr(cfg, "tdt") or hasattr(cfg, "model_defaults")):
        defaults = getattr(cfg, "model_defaults", None)
        if defaults is not None and hasattr(defaults, "tdt_durations"):
            return "tdt"
    joint = getattr(model, "joint", None)
    if joint is not None and (hasattr(joint, "duration_head") or hasattr(joint, "dur_pred")):
        return "tdt"
    return "rnnt"


def build_config(model: Any, decoder_type: str) -> Dict[str, Any]:
    cfg = model.cfg
    config: Dict[str, Any] = {
        "encoder_layers": int(cfg.encoder.n_layers),
        "d_model": int(cfg.encoder.d_model),
        "num_heads": int(cfg.encoder.n_heads),
        "conv_kernel_size": int(cfg.encoder.conv_kernel_size),
        "ff_expansion_factor": int(cfg.encoder.ff_expansion_factor),
        "subsampling_factor": int(getattr(cfg.encoder, "subsampling_factor", 8)),
        "vocab_size": int(model.decoder.prediction.vocab_size),
        "pred_hidden": int(model.decoder.prediction.hidden_size),
        "num_mel_bands": int(cfg.preprocessor.features),
        "sample_rate": int(cfg.preprocessor.sample_rate),
        "window_size_ms": float(cfg.preprocessor.window_size) * 1000.0,
        "hop_size_ms": float(getattr(cfg.preprocessor, "hop_length_ms", 10.0)),
        "att_context_size": _to_plain(getattr(cfg.encoder, "att_context_size", [70, 1])),
        "decoder_type": decoder_type,
        "blank_id": 0,
    }
    defaults = getattr(cfg, "model_defaults", None)
    if decoder_type == "tdt" and defaults is not None and hasattr(defaults, "tdt_durations"):
        durations = _to_plain(defaults.tdt_durations)
        config["tdt_durations"] = durations
        config["tdt_num_durations"] = len(durations)
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Export NeMo checkpoint to SafeTensors")
    parser.add_argument("--model", required=True, help="NeMo pretrained model name")
    parser.add_argument("--out-dir", required=True, help="Directory for export artifacts")
    parser.add_argument(
        "--safetensors-name",
        default="model.safetensors",
        help="SafeTensors output filename (default: model.safetensors)",
    )
    args = parser.parse_args()

    try:
        import nemo.collections.asr as nemo_asr
        from safetensors.torch import save_file
    except Exception as exc:
        raise RuntimeError(
            "Missing dependencies. Install nemo_toolkit[asr] and safetensors in your Python environment."
        ) from exc

    os.makedirs(args.out_dir, exist_ok=True)
    model = nemo_asr.models.ASRModel.from_pretrained(args.model)

    state_dict = {key: tensor.contiguous().cpu() for key, tensor in model.state_dict().items()}
    safetensors_path = os.path.join(args.out_dir, args.safetensors_name)
    save_file(state_dict, safetensors_path)

    decoder_type = infer_decoder_type(model)
    config = build_config(model, decoder_type)
    config_path = os.path.join(args.out_dir, "model_config.json")
    with open(config_path, "w", encoding="utf-8") as file:
        json.dump(config, file, indent=2, ensure_ascii=False)

    print(f"Exported {len(state_dict)} tensors to {safetensors_path}")
    print(f"Wrote config to {config_path}")


if __name__ == "__main__":
    main()
