#!/usr/bin/env python3
"""
Export a NeMo ASR checkpoint to SafeTensors + model_config JSON for audio-ml ASR.

Requires: pip install nemo_toolkit safetensors torch

Example:
  python tools/export_nemo_to_safetensors.py --model nvidia/parakeet_realtime_eou_120m-v1 --out ./exports
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="NeMo model id or local .nemo path")
    p.add_argument("--out", default="exports", help="Output directory")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    try:
        import nemo.collections.asr as nemo_asr
        from safetensors.torch import save_file
    except ImportError as e:
        raise SystemExit(
            "Install NeMo + safetensors: pip install nemo_toolkit safetensors torch\n" + str(e)
        ) from e

    model = nemo_asr.models.ASRModel.from_pretrained(args.model)
    state_dict = {k: v.contiguous() for k, v in model.state_dict().items()}
    save_file(state_dict, str(out / "model.safetensors"))

    decoder_type = "tdt" if (hasattr(model.cfg, "tdt") or hasattr(model.joint, "duration")) else "rnnt"
    cfg = {
        "encoder_layers": model.cfg.encoder.n_layers,
        "d_model": model.cfg.encoder.d_model,
        "num_heads": model.cfg.encoder.n_heads,
        "conv_kernel_size": model.cfg.encoder.conv_kernel_size,
        "ff_expansion_factor": model.cfg.encoder.ff_expansion_factor,
        "subsampling_factor": model.cfg.encoder.subsampling_factor,
        "subsampling_conv_channels": getattr(model.cfg.encoder, "subsampling_conv_channels", model.cfg.encoder.d_model),
        "vocab_size": model.decoder.prediction.vocab_size,
        "pred_hidden": model.decoder.prediction.hidden_size,
        "num_mel_bands": model.cfg.preprocessor.features,
        "sample_rate": model.cfg.preprocessor.sample_rate,
        "window_size_ms": model.cfg.preprocessor.window_size * 1000,
        "hop_size_ms": model.cfg.preprocessor.hop_length_ms,
        "att_context_size": list(model.cfg.encoder.att_context_size),
        "decoder_type": decoder_type,
        "self_attention_model": getattr(model.cfg.encoder, "self_attention_model", "rel_pos"),
        "xscale": getattr(model.cfg.encoder, "xscale", True),
        "state_dict_prefix": "",
        "decoder_state_dict_prefix": "",
    }
    if decoder_type == "tdt" and hasattr(model.cfg, "model_defaults"):
        md = model.cfg.model_defaults
        if hasattr(md, "tdt_durations"):
            cfg["tdt_durations"] = list(md.tdt_durations)

    (out / "model_config.json").write_text(json.dumps(cfg, indent=2))
    keys = list(state_dict.keys())[:200]
    (out / "state_dict_keys_sample.txt").write_text("\n".join(keys))
    print(f"Wrote {out / 'model.safetensors'} and model_config.json")
    print("Tip: export SentencePiece vocab separately (export_vocab.py).")


if __name__ == "__main__":
    main()
