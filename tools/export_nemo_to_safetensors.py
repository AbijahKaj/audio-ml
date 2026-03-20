#!/usr/bin/env python3
"""Export a NeMo FastConformer checkpoint into SafeTensors + JSON config."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nemo.collections.asr as nemo_asr
from safetensors.torch import save_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="NeMo/Hugging Face model name")
    parser.add_argument("--output-dir", default="exported-model", help="Directory for exported artifacts")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = nemo_asr.models.ASRModel.from_pretrained(args.model)
    state_dict = {name: tensor.contiguous() for name, tensor in model.state_dict().items()}
    save_file(state_dict, str(output_dir / "model.safetensors"))

    decoder_type = "tdt" if hasattr(model.joint, "duration_head") else "rnnt"
    config = {
        "encoder_layers": int(model.cfg.encoder.n_layers),
        "d_model": int(model.cfg.encoder.d_model),
        "num_heads": int(model.cfg.encoder.n_heads),
        "conv_kernel_size": int(model.cfg.encoder.conv_kernel_size),
        "ff_expansion_factor": int(model.cfg.encoder.ff_expansion_factor),
        "subsampling_factor": int(model.cfg.encoder.subsampling_factor),
        "vocab_size": int(model.decoder.prediction.vocab_size),
        "pred_hidden": int(model.decoder.prediction.hidden_size),
        "num_mel_bands": int(model.cfg.preprocessor.features),
        "sample_rate": int(model.cfg.preprocessor.sample_rate),
        "window_size_ms": float(model.cfg.preprocessor.window_size) * 1000.0,
        "hop_size_ms": float(model.cfg.preprocessor.window_stride) * 1000.0,
        "att_context_size": list(model.cfg.encoder.att_context_size),
        "decoder_type": decoder_type,
    }

    if decoder_type == "tdt":
        durations = getattr(model.cfg.model_defaults, "tdt_durations", None)
        if durations is not None:
          config["tdt_durations"] = list(durations)
          config["tdt_num_durations"] = len(durations)

    with (output_dir / "model_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)


if __name__ == "__main__":
    main()
