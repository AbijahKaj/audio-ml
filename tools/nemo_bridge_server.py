#!/usr/bin/env python3
"""
Persistent JSON-lines bridge for NeMo ASR inference.

Input (stdin, one JSON per line):
  {"id": 1, "cmd": "transcribe", "sample_rate": 16000, "pcm": [0.0, ...]}
  {"id": 2, "cmd": "shutdown"}

Output (stdout, one JSON per line):
  {"event":"ready","sample_rate":16000,"decoder_type":"rnnt","model_name":"..."}
  {"id":1,"ok":true,"text":"..."}
  {"id":2,"ok":true}
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import traceback
from typing import Any, Dict, Iterable

import numpy as np
import soundfile as sf


def _json_line(payload: Dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _resample_linear(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate or audio.size == 0:
        return audio.astype(np.float32, copy=False)

    ratio = dst_rate / src_rate
    out_len = int(round(audio.shape[0] * ratio))
    if out_len <= 0:
        return np.zeros((0,), dtype=np.float32)

    positions = np.arange(out_len, dtype=np.float32) / ratio
    left = np.floor(positions).astype(np.int64)
    right = np.clip(left + 1, 0, audio.shape[0] - 1)
    frac = positions - left.astype(np.float32)
    left = np.clip(left, 0, audio.shape[0] - 1)

    out = (1.0 - frac) * audio[left] + frac * audio[right]
    return out.astype(np.float32)


def _infer_decoder_type(model: Any) -> str:
    joint = getattr(model, "joint", None)
    if joint is not None and (hasattr(joint, "duration_head") or hasattr(joint, "dur_pred")):
        return "tdt"
    cfg = getattr(model, "cfg", None)
    if cfg is not None and hasattr(cfg, "tdt"):
        return "tdt"
    return "rnnt"


def _extract_text(item: Any) -> str:
    if hasattr(item, "text"):
        return str(item.text)
    if isinstance(item, str):
        return item
    return str(item)


class NemoBridgeServer:
    def __init__(self, model_name: str):
        import nemo.collections.asr as nemo_asr

        self.model = nemo_asr.models.ASRModel.from_pretrained(model_name)
        self.model.eval()
        self.model_name = model_name
        self.sample_rate = int(self.model.cfg.preprocessor.sample_rate)
        self.decoder_type = _infer_decoder_type(self.model)

    def ready_payload(self) -> Dict[str, Any]:
        return {
            "event": "ready",
            "sample_rate": self.sample_rate,
            "decoder_type": self.decoder_type,
            "model_name": self.model_name,
        }

    def transcribe(self, pcm: Iterable[float], sample_rate: int) -> str:
        audio = np.asarray(list(pcm), dtype=np.float32)
        audio = _resample_linear(audio, int(sample_rate), self.sample_rate)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as file:
            tmp_path = file.name
        try:
            sf.write(tmp_path, audio, self.sample_rate, subtype="PCM_16")
            result = self.model.transcribe([tmp_path], batch_size=1)
            if not result:
                return ""
            return _extract_text(result[0])
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="NeMo ASR bridge server")
    parser.add_argument("--model", required=True, help="NeMo model name")
    args = parser.parse_args()

    server = NemoBridgeServer(args.model)
    _json_line(server.ready_payload())

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            req_id = int(request.get("id", -1))
            cmd = request.get("cmd")

            if cmd == "shutdown":
                _json_line({"id": req_id, "ok": True})
                return

            if cmd == "transcribe":
                sample_rate = int(request.get("sample_rate", server.sample_rate))
                pcm = request.get("pcm", [])
                text = server.transcribe(pcm, sample_rate)
                _json_line({"id": req_id, "ok": True, "text": text})
                continue

            _json_line({"id": req_id, "ok": False, "error": f"Unknown command: {cmd}"})
        except Exception as exc:  # pragma: no cover - defensive bridge safety
            _json_line(
                {
                    "id": int(request.get("id", -1)) if "request" in locals() else -1,
                    "ok": False,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )


if __name__ == "__main__":
    main()
