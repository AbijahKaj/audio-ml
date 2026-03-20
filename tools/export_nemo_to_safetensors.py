#!/usr/bin/env python3
"""
Export a NeMo FastConformer ASR checkpoint to SafeTensors + model_config.json.

Requires: pip install nemo_toolkit safetensors torch

Example:
  python export_nemo_to_safetensors.py --model nvidia/parakeet_realtime_eou_120m-v1 --out parakeet_120m
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="NeMo model name or .nemo path")
    parser.add_argument("--out", required=True, help="Output prefix (without extension)")
    args = parser.parse_args()

    import nemo.collections.asr as nemo_asr
    from safetensors.torch import save_file

    model = nemo_asr.models.ASRModel.restore_from(args.model) if args.model.endswith(
        ".nemo"
    ) else nemo_asr.models.ASRModel.from_pretrained(args.model)

    state_dict = {k: v.contiguous() for k, v in model.state_dict().items()}
    out_path = Path(args.out).with_suffix(".safetensors")
    save_file(state_dict, str(out_path))

    decoder_type = "tdt"
    if not (hasattr(model.cfg, "tdt") or hasattr(getattr(model, "joint", None), "duration_head")):
        decoder_type = "rnnt"

    enc = model.cfg.encoder
    prep = model.cfg.preprocessor
    joint = model.cfg.joint
    pred_hidden = model.cfg.model_defaults.pred_hidden
    enc_hidden = model.cfg.model_defaults.enc_hidden
    joint_hidden = joint.jointnet.joint_hidden
    joint_act = str(joint.jointnet.activation).lower()

    tdt_durations = None
    tdt_num = None
    if decoder_type == "tdt" and hasattr(model.cfg, "model_defaults"):
        md = model.cfg.model_defaults
        if hasattr(md, "tdt_durations"):
            tdt_durations = list(md.tdt_durations)
            tdt_num = len(tdt_durations)

    config = {
        "encoder_layers": enc.n_layers,
        "d_model": enc.d_model,
        "num_heads": enc.n_heads,
        "conv_kernel_size": enc.conv_kernel_size,
        "ff_expansion_factor": enc.ff_expansion_factor,
        "subsampling_factor": enc.subsampling_factor,
        "vocab_size": model.decoder.prediction.embed.num_embeddings - 1,
        "pred_hidden": pred_hidden,
        "enc_hidden": enc_hidden,
        "joint_hidden": joint_hidden,
        "joint_activation": joint_act,
        "num_mel_bands": prep.features,
        "sample_rate": prep.sample_rate,
        "window_size_ms": float(prep.window_size) * 1000.0,
        "hop_size_ms": float(getattr(prep, "hop_length_ms", 10)),
        "att_context_size": list(enc.att_context_size),
        "decoder_type": decoder_type,
        "self_attention_model": enc.self_attention_model,
        "pos_emb_max_len": enc.pos_emb_max_len,
        "xscale": bool(enc.xscaling) if hasattr(enc, "xscaling") else True,
        "subsampling": enc.subsampling,
        "subsampling_conv_channels": enc.subsampling_conv_channels,
        "blank_token_id": model.decoder.blank_idx,
    }
    if tdt_durations is not None:
        config["tdt_durations"] = tdt_durations
        config["tdt_num_durations"] = tdt_num

    cfg_path = Path(args.out).with_name(Path(args.out).name + "_config.json")
    cfg_path.write_text(json.dumps(config, indent=2))
    print(f"Wrote {out_path} and {cfg_path}")


if __name__ == "__main__":
    main()
