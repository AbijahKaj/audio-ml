"""
Export a NeMo FastConformer ASR model to SafeTensors format for use with audio-ml.

Usage:
    pip install nemo_toolkit[asr] safetensors torch
    python export_nemo_to_safetensors.py --model nvidia/parakeet_realtime_eou_120m-v1 --out parakeet_120m

Supported models:
    - nvidia/parakeet_realtime_eou_120m-v1   (RNNT, 120M, streaming)
    - nvidia/stt_en_fastconformer_tdt_large   (TDT, 115M)
    - nvidia/parakeet-tdt_ctc-110m            (TDT+CTC, 110M)
    - nvidia/nemotron-speech-streaming-en-0.6b (RNNT, 600M, streaming)
    - nvidia/parakeet-tdt-0.6b-v3             (TDT multilingual, 600M)
"""

import argparse
import json
import os

import torch
import nemo.collections.asr as nemo_asr
from safetensors.torch import save_file


def get_decoder_type(model) -> str:
    """Detect whether the model uses RNNT or TDT decoder."""
    if hasattr(model, "joint") and hasattr(model.joint, "duration_head"):
        return "tdt"
    if hasattr(model, "cfg"):
        cfg = model.cfg
        if hasattr(cfg, "tdt") or (hasattr(cfg, "model_defaults") and
                                    hasattr(cfg.model_defaults, "tdt_durations")):
            return "tdt"
    return "rnnt"


def get_tdt_durations(model) -> list:
    """Extract TDT duration bins from model config."""
    try:
        return list(model.cfg.model_defaults.tdt_durations)
    except AttributeError:
        pass
    try:
        return list(model.cfg.tdt.durations)
    except AttributeError:
        pass
    return [0, 1, 2, 3, 4]


def export_model(model_name: str, out_prefix: str):
    print(f"Loading model: {model_name}")
    model = nemo_asr.models.ASRModel.from_pretrained(model_name)
    model.eval()

    # Export weights as SafeTensors
    weights_path = f"{out_prefix}.safetensors"
    print(f"Exporting weights to {weights_path}")
    state_dict = {k: v.contiguous().cpu() for k, v in model.state_dict().items()}
    save_file(state_dict, weights_path)
    print(f"  -> {len(state_dict)} tensors saved")

    # Print all weight names for reference (useful for building WeightMapper)
    keys_path = f"{out_prefix}_weight_keys.txt"
    with open(keys_path, "w") as f:
        for k, v in state_dict.items():
            f.write(f"{k}  {list(v.shape)}\n")
    print(f"  -> Weight keys written to {keys_path}")

    # Build config
    decoder_type = get_decoder_type(model)
    cfg = model.cfg

    config = {
        "encoder_layers": int(cfg.encoder.n_layers),
        "d_model": int(cfg.encoder.d_model),
        "num_heads": int(cfg.encoder.n_heads),
        "conv_kernel_size": int(cfg.encoder.conv_kernel_size),
        "ff_expansion_factor": int(cfg.encoder.ff_expansion_factor),
        "subsampling_factor": int(getattr(cfg.encoder, "subsampling_factor", 8)),
        "subsampling_conv_channels": int(getattr(cfg.encoder, "subsampling_conv_channels", 256)),
        "vocab_size": int(model.decoder.prediction.vocab_size),
        "pred_hidden": int(model.decoder.prediction.hidden_size),
        "pred_embed_dim": int(getattr(model.decoder.prediction, "embed_dim", 128)),
        "num_mel_bands": int(cfg.preprocessor.features),
        "sample_rate": int(cfg.preprocessor.sample_rate),
        "window_size_ms": float(cfg.preprocessor.window_size) * 1000,
        "hop_size_ms": float(getattr(cfg.preprocessor, "hop_length_ms",
                                      getattr(cfg.preprocessor, "window_stride", 0.01) * 1000)),
        "att_context_size": list(getattr(cfg.encoder, "att_context_size", [70, 1])),
        "decoder_type": decoder_type,
    }

    if decoder_type == "tdt":
        durations = get_tdt_durations(model)
        config["tdt_durations"] = durations
        config["tdt_num_durations"] = len(durations)

    config_path = f"{out_prefix}_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  -> Config written to {config_path}")
    print(f"     decoder_type = {decoder_type}")
    print(f"     encoder_layers = {config['encoder_layers']}")
    print(f"     d_model = {config['d_model']}")
    print(f"     vocab_size = {config['vocab_size']}")

    print("Done.")
    return model, config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="nvidia/parakeet_realtime_eou_120m-v1")
    parser.add_argument("--out", default="parakeet_120m")
    args = parser.parse_args()
    export_model(args.model, args.out)
