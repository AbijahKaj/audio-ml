# NeMo vs TypeScript encoder parity

## Environment (reference)

```bash
pip install 'numpy<2' torch nemo_toolkit[asr]==2.0.0 soundfile safetensors sentencepiece matplotlib
```

## Generate fixtures

```bash
python3 tools/export_parakeet_fixture.py --out tmp_parakeet
```

Produces:

- `model.safetensors` — full checkpoint
- `model_config.json` — fields aligned with `parseModelConfig`
- `mel_bt_f.bin` / `mel_shape.json` — preprocessor features as `[B, T, F]` (same layout as NeMo after `transpose(1,2)` into the encoder)
- `encoder_ref_btd.bin` — NeMo `encoder()` output as `[B, T, d_model]`

## What was verified

1. **Parakeet Realtime EOU 120M** loads in NeMo 2.0 and exports to SafeTensors.
2. **Weight keys** differ slightly from the older plan (`decoder.prediction.dec_rnn.lstm.*`, conv `batch_norm` is LayerNorm, no `conv.norm`).
3. **Subsampling** must use **`dw_striding`** (depthwise + pointwise blocks), not plain striding.
4. **Mel layout** into convs is **`[B, 1, T, F]`** (time × mel), not `[B, 1, F, T]`.

## Current numerical gap

On the same `mel_bt_f` input, the TS subsampling path still yields a **different 2D shape** before `encoder.pre_encode.out` than PyTorch (e.g. 13×16 vs 14×17 feature grid). That comes from **conv2d padding / rounding** differences between TF.js `same` and PyTorch `padding=1` over three stride-2 stages on odd lengths (`T=101`, `F=128`).

Closing the gap requires either:

- per-layer explicit padding copied from NeMo’s `calc_length` / conv definitions, or  
- running the subsampling block in Python/ONNX and only porting the Conformer stack in TF.js.

## Optional Vitest smoke test

```bash
yarn vitest run src/asr/integration/parakeetEncoderParity.test.ts
```

The integration test loads the real checkpoint and runs the TS encoder; full bitwise parity is **not** asserted until padding is aligned.
