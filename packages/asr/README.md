# @audio-ml/asr

FastConformer speech recognition in **TypeScript**, powered by **TensorFlow.js**. Models are exported from [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) to SafeTensors plus JSON config and vocabulary—see the [`audio-ml` repo](https://github.com/AbijahKaj/audio-ml) and `tools/export_nemo_to_safetensors.py`.

Uses a **TDT (Token-and-Duration Transducer)** decoder for both streaming and offline transcription. TDT predicts both a token and a duration (frame skip), enabling 2–5× faster decoding than traditional RNN-T.

This package depends on **[`audio-ml`](https://www.npmjs.com/package/audio-ml)** for shared application types (for example `BaseApplication` and VAD used by streaming endpointing).

## Install

```bash
npm install audio-ml @audio-ml/asr
```

Optional, for **native TensorFlow in Node.js** (faster than pure JS CPU):

```bash
npm install @tensorflow/tfjs-node
```

`@tensorflow/tfjs-node` is an **optional peer**; install it only when you use the `tensorflow` backend in Node.

## Ready-to-use models (same as the `audio-ml` demo)

These Hugging Face repos ship `model.safetensors`, `model_config.json`, and `vocab.json` on the `main` branch (NeMo → export via `tools/export_nemo_to_safetensors.py` in the main repo).

| Model | Hugging Face repo | Notes |
|-------|-------------------|--------|
| Parakeet TDT 110M | [AbijahKaj/parakeet-tdt-110m-web](https://huggingface.co/AbijahKaj/parakeet-tdt-110m-web) | English, TDT decoder, ~220 MB weights |
| FastConformer TDT Large | [AbijahKaj/fastconformer-tdt-large-web](https://huggingface.co/AbijahKaj/fastconformer-tdt-large-web) | English, TDT, ~218 MB weights |
| Parakeet TDT 0.6B v3 | [AbijahKaj/parakeet-tdt-0.6b-v3-web](https://huggingface.co/AbijahKaj/parakeet-tdt-0.6b-v3-web) | **Multilingual** (25 European languages), TDT decoder, punctuation & capitalization, auto language detection, ~1.2 GB float16 weights — use the `webgpu` backend |

### Multilingual (Parakeet TDT 0.6B v3)

[`nvidia/parakeet-tdt-0.6b-v3`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) is a
600M-parameter FastConformer-TDT model covering **25 European languages** (English, French,
German, Spanish, Italian, Portuguese, Dutch, Polish, Russian, Ukrainian, and 15 more) with
automatic language detection, punctuation, and capitalization. It shares the exact same
FastConformer encoder + TDT decoder architecture as the English models, so this package runs
it unchanged — it only needs the v3 weights and its larger SentencePiece vocabulary.

The vocabulary uses **byte-fallback** tokens (`<0xNN>`) for characters outside the subword
vocabulary; `SentencePieceDecoder` decodes consecutive byte tokens back into the correct
UTF-8 characters, so accented and non-Latin scripts transcribe correctly.

Export it yourself from the NeMo checkpoint:

```bash
python tools/export_nemo_to_safetensors.py \
  --model nvidia/parakeet-tdt-0.6b-v3 \
  --output-dir exported/parakeet-tdt-0.6b-v3
```

> The float16 weights are ~1.2 GB. Prefer the `webgpu` backend; CPU/WASM are impractical at
> this size.

Resolve URLs follow this pattern (`{repo}` = `username/repo`):

`https://huggingface.co/{repo}/resolve/main/model.safetensors`  
`https://huggingface.co/{repo}/resolve/main/model_config.json`  
`https://huggingface.co/{repo}/resolve/main/vocab.json`

## Quick start

Example using **Parakeet TDT 110M** (same default-style URLs as `demo/pages/SpeechRecognizerDemo.ts`):

```typescript
import { FastConformerASR, type ASRResult } from '@audio-ml/asr';

const HF = 'https://huggingface.co/AbijahKaj/parakeet-tdt-110m-web/resolve/main';

const asr = new FastConformerASR({
  sampleRate: 16_000,
  modelPath: `${HF}/model.safetensors`,
  configPath: `${HF}/model_config.json`,
  vocabPath: `${HF}/vocab.json`,
  backend: 'webgpu', // browser: 'webgpu' | 'webgl' | 'wasm' | 'cpu'
  streaming: true,
});

await asr.load();

asr.on('partial', (p) => console.log(p.text));
asr.on('final', (r: ASRResult) => console.log(r.text));

asr.processFrame(pcmFrame);
```

To load from already-fetched buffers:

```typescript
await asr.loadFromBuffers(modelArrayBuffer, configJsonString, vocabJsonString);
```

Offline pass (full audio transcription):

```typescript
const result = await asr.transcribe(audioFloat32);
```

Streaming inference with VAD endpointing:

```typescript
// Construct with streaming + VAD enabled
const asr = new FastConformerASR({
  sampleRate: 16_000,
  modelPath: `${HF}/model.safetensors`,
  configPath: `${HF}/model_config.json`,
  vocabPath: `${HF}/vocab.json`,
  backend: 'webgpu',
  streaming: true,
  chunkSizeMs: 2000,
  silenceTimeoutMs: 1200,
});

// processFrame feeds audio and fires 'partial' / 'final' events
asr.on('partial', (p) => setLivePreview(p.text));
asr.on('final', (r: ASRResult) => appendTranscript(r.text));
asr.processFrame(pcmFrame);
```

## TensorFlow.js backends

| Backend       | Typical use |
|---------------|-------------|
| `webgpu`      | Browser, best GPU path when supported |
| `webgl`       | Browser, broader GPU support |
| `wasm`        | Browser, good CPU throughput via WASM |
| `cpu`         | Browser or Node, pure JS (slow for large models) |
| `tensorflow`  | **Node only** — requires `@tensorflow/tfjs-node` |

WASM backend options:

```typescript
await asr.load(); // after constructing with:
// backend: 'wasm',
// backendOptions: { wasmPathPrefix: '/tfjs-wasm/' }
```

Serve `.wasm` files from `tfjs-backend-wasm` with correct MIME type (see the main repo demo Vite config).

## Swappable compute layer

Inference is expressed against a **`ComputeBackend`** interface. **`TfjsBackend`** is the default implementation; you can supply another backend that implements the same operations if you integrate a different runtime.

## Exports

Besides **`FastConformerASR`**, the package exports encoder/decoder/feature/text/model helpers (for example `FastConformerEncoder`, `createDecoder`, `FeaturePipeline`, `loadSafeTensors`, `parseModelConfig`, streaming types, and `Endpointer`). See `src/index.ts` for the full public API.

## Requirements

- **Node.js** ≥ 18
- **Peer:** `audio-ml` ^1.0.0

## License

MIT — see [LICENSE](./LICENSE).

## Repository

[github.com/AbijahKaj/audio-ml](https://github.com/AbijahKaj/audio-ml) (package path: `packages/asr`).
