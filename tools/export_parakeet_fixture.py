#!/usr/bin/env python3
"""Export Parakeet 120M weights + mel (B,T,F) + NeMo encoder reference for TS parity tests."""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="tmp_parakeet", help="Output directory")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import nemo.collections.asr as nemo_asr
    from safetensors.torch import save_file

    model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet_realtime_eou_120m-v1")
    model.eval()

    state_dict = {k: v.contiguous() for k, v in model.state_dict().items()}
    save_file(state_dict, str(out / "model.safetensors"))

    enc = model.cfg.encoder
    prep = model.cfg.preprocessor
    joint = model.cfg.joint
    cfg = {
        "encoder_layers": enc.n_layers,
        "d_model": enc.d_model,
        "num_heads": enc.n_heads,
        "conv_kernel_size": enc.conv_kernel_size,
        "ff_expansion_factor": enc.ff_expansion_factor,
        "subsampling_factor": enc.subsampling_factor,
        "vocab_size": model.decoder.prediction.embed.num_embeddings - 1,
        "pred_hidden": model.cfg.model_defaults.pred_hidden,
        "enc_hidden": model.cfg.model_defaults.enc_hidden,
        "joint_hidden": joint.jointnet.joint_hidden,
        "joint_activation": str(joint.jointnet.activation).lower().replace("relu()", "relu"),
        "num_mel_bands": prep.features,
        "sample_rate": prep.sample_rate,
        "window_size_ms": float(prep.window_size) * 1000.0,
        "hop_size_ms": float(getattr(prep, "hop_length_ms", 10)),
        "att_context_size": list(enc.att_context_size),
        "decoder_type": "rnnt",
        "self_attention_model": enc.self_attention_model,
        "pos_emb_max_len": enc.pos_emb_max_len,
        "xscale": bool(getattr(enc, "xscaling", True)),
        "subsampling": enc.subsampling,
        "subsampling_conv_channels": enc.subsampling_conv_channels,
        "blank_token_id": int(model.decoder.blank_idx),
    }
    (out / "model_config.json").write_text(json.dumps(cfg, indent=2))

    torch.manual_seed(42)
    sig = torch.randn(1, 16000, dtype=torch.float32) * 0.02
    lens = torch.tensor([16000], dtype=torch.int64)
    with torch.no_grad():
        feats, fl = model.preprocessor(input_signal=sig, length=lens)
        enc_out, el = model.encoder(audio_signal=feats, length=fl)

    # NeMo feats [B, F, T] -> TS mel [B, T, F]
    mel_btf = np.transpose(feats.cpu().numpy(), (0, 2, 1)).astype(np.float32)
    mel_btf.tofile(out / "mel_bt_f.bin")
    (out / "mel_shape.json").write_text(json.dumps(list(mel_btf.shape)))

    # NeMo encoder [B, D, T] -> TS [B, T, D]
    enc_btd = np.transpose(enc_out.cpu().numpy(), (0, 2, 1)).astype(np.float32)
    enc_btd.tofile(out / "encoder_ref_btd.bin")
    (out / "encoder_ref_shape.json").write_text(json.dumps(list(enc_btd.shape)))

    print("Wrote", out)
    print("mel", mel_btf.shape, "enc_ref", enc_btd.shape)


if __name__ == "__main__":
    main()
