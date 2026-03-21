## Plan: FastConformer ASR Engine with TensorFlow.js

Supports both **RNNT** (RNN-Transducer) and **TDT** (Token-and-Duration Transducer) decoder variants. The encoder is shared — the only difference is the joint network output structure and decode loop.

- **RNNT models**: Parakeet 120M, Nemotron 0.6B, and other NeMo FastConformer-RNNT checkpoints
- **TDT models**: NeMo FastConformer-TDT checkpoints (joint network outputs token logits + duration logits, enabling frame-skipping for faster inference)

---

### References & Reading List

#### Papers (read in this order)

| # | Paper | What It Covers | Relevance |
|---|---|---|---|
| 1 | [Conformer: Convolution-augmented Transformer for Speech Recognition](https://arxiv.org/abs/2005.08100) (Gulati et al., 2020) | Original Conformer architecture — Macaron FFN sandwich, multi-head self-attention + convolution module | Understand the base architecture before FastConformer's modifications |
| 2 | [Fast Conformer with Linearly Scalable Attention for Efficient Speech Recognition](https://arxiv.org/abs/2305.05084) (Rekesh et al., 2023) | FastConformer — 8x downsampling, limited context attention, scaling to 1B+ params | The encoder architecture being implemented |
| 3 | [Sequence Transduction with Recurrent Neural Networks](https://arxiv.org/abs/1211.3711) (Graves, 2012) | RNN-Transducer — encoder + prediction network + joint network, transducer loss | The RNNT decoder framework |
| 4 | [Connectionist Temporal Classification](https://www.cs.toronto.edu/~graves/icml_2006.pdf) (Graves et al., 2006) | CTC — blank tokens, alignment-free training, greedy/beam decoding | Background for understanding how CTC relates to RNNT |
| 5 | [Efficient Sequence Transduction by Jointly Predicting Tokens and Durations](https://arxiv.org/abs/2304.06795) (Xu et al., ICML 2023) | TDT — duration prediction head, frame-skipping decode, 2-5x faster than RNNT | The TDT decoder being implemented |
| 6 | [Stateful Conformer with Cache-based Inference for Streaming ASR](https://arxiv.org/abs/2312.17279) (Noroozi et al., 2023) | Cache-aware streaming — activation caching for attention KV and conv states, chunked inference | The streaming mechanism (Phase 7) |
| 7 | [Relative Positional Encoding for Speech Recognition and Direct Translation](https://www.isca-archive.org/interspeech_2020/pham20_interspeech.html) (Pham et al., Interspeech 2020) | Relative position encoding in self-attention for speech — why it outperforms absolute | Understanding the attention module's position encoding |
| 8 | [SentencePiece: A simple and language independent subword tokenizer](https://arxiv.org/abs/1808.06226) (Kudo & Richardson, 2018) | BPE/unigram subword tokenization — how token vocabularies are built | The tokenizer format — this project implements the decode side |
| 9 | [Canary-1B-v2 & Parakeet-TDT-0.6B-v3: Efficient and High-Performance Models for Multilingual ASR and AST](https://arxiv.org/abs/2509.14128) (NVIDIA, 2025) | Multilingual training, auto language detection, Parakeet v3 architecture details | Architecture details for the multilingual model used for French testing |

#### Hugging Face Models

**English-only models** (start here for development):

| Model | Type | Params | Use Case | Link |
|---|---|---|---|---|
| Parakeet Realtime EOU 120M | FastConformer-RNNT, streaming, EOU detection | 120M | Primary dev target — small, browser-viable | [nvidia/parakeet_realtime_eou_120m-v1](https://huggingface.co/nvidia/parakeet_realtime_eou_120m-v1) |
| Nemotron Speech Streaming 0.6B | FastConformer-RNNT, cache-aware streaming, PnC | 600M | Node.js server target — high accuracy | [nvidia/nemotron-speech-streaming-en-0.6b](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b) |
| FastConformer TDT Large | FastConformer-TDT | 115M | TDT decoder validation | [nvidia/stt_en_fastconformer_tdt_large](https://huggingface.co/nvidia/stt_en_fastconformer_tdt_large) |
| Parakeet TDT-CTC 110M | FastConformer-TDT + CTC hybrid | 114M | Small TDT model, browser-viable | [nvidia/parakeet-tdt_ctc-110m](https://huggingface.co/nvidia/parakeet-tdt_ctc-110m) |
| Parakeet TDT 1.1B | FastConformer-TDT | 1.1B | Highest accuracy English TDT model | [nvidia/parakeet-tdt-1.1b](https://huggingface.co/nvidia/parakeet-tdt-1.1b) |

**Multilingual models** (French + 24 other languages):

| Model | Type | Params | Languages | Use Case | Link |
|---|---|---|---|---|---|
| **Parakeet TDT 0.6B v3** | FastConformer-TDT, multilingual, PnC, timestamps | 600M | 25 (EN, **FR**, DE, ES, PT, IT, NL, PL, RU, UK, + 15 more) | **Primary multilingual/French target** — same encoder architecture, TDT decoder, auto language detection | [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) |
| Canary 1B v2 | FastConformer encoder + Transformer decoder, ASR + translation | 918M | 25+ (EN, **FR**, DE, ES, + more) | Multilingual ASR + speech translation (FR↔EN), higher accuracy but different decoder architecture (autoregressive Transformer, not transducer) | [nvidia/canary-1b-v2](https://huggingface.co/nvidia/canary-1b-v2) |
| Parakeet TDT 0.6B v3 (FR fine-tune) | Fine-tuned for French media (radio, TV, podcasts) | 600M | FR (optimized), other langs degraded | French-specific use cases — 57-60% WER improvement on French news/documentary vs base model | [Archime/parakeet-tdt-0.6b-v3-fr-tv-media](https://huggingface.co/Archime/parakeet-tdt-0.6b-v3-fr-tv-media) |

> **Note on Parakeet TDT 0.6B v3 for French**: This is the most practical multilingual target because it uses the exact same FastConformer-TDT architecture as the English models. The engine runs it with zero code changes — just different weights and a larger SentencePiece vocabulary. It supports automatic language detection, punctuation, capitalization, and word-level timestamps. The Canary 1B v2 uses a Transformer decoder (autoregressive, not transducer), which would require implementing an additional decoder type.

#### Documentation & Code

| Resource | What It Covers | Link |
|---|---|---|
| NeMo ASR Documentation | Model configs, training, inference, checkpoint loading | [NeMo ASR User Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/intro.html) |
| NeMo FastConformer Config | Full YAML config with all hyperparameters (layers, heads, kernel sizes) | [fast-conformer_transducer_bpe.yaml](https://github.com/NVIDIA/NeMo/blob/main/examples/asr/conf/fastconformer/fast-conformer_transducer_bpe.yaml) |
| NeMo Cache-Aware Streaming Script | Reference implementation for streaming inference | [speech_to_text_cache_aware_streaming_infer.py](https://github.com/NVIDIA/NeMo/blob/main/examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py) |
| NeMo Source (ConformerEncoder) | Python source for the encoder to port to JS | [NeMo GitHub — modules/conformer_encoder.py](https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/asr/modules/conformer_encoder.py) |
| NeMo Source (RNNT Decoder) | Python source for the RNNT decoder | [NeMo GitHub — modules/rnnt.py](https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/asr/modules/rnnt.py) |
| SafeTensors Format Spec | File format for weight loading (header + binary layout) | [SafeTensors Docs](https://huggingface.co/docs/safetensors) / [GitHub](https://github.com/huggingface/safetensors) |
| TensorFlow.js Guide | Core API (tensors, ops, backends, memory management) | [TensorFlow.js Guide](https://www.tensorflow.org/js/guide) |
| TensorFlow.js npm | Installation, versions, backend packages | [@tensorflow/tfjs on npm](https://www.npmjs.com/package/@tensorflow/tfjs) |
| SentencePiece | Tokenizer source code and pretrained models | [google/sentencepiece on GitHub](https://github.com/google/sentencepiece) |
| WebGPU Compute (for future backend) | GPU compute shaders, WGSL language, getting started | [Chrome WebGPU Compute Guide](https://web.dev/gpu-compute/) / [Google Codelab](https://codelabs.developers.google.com/your-first-webgpu-app) |

#### Background Concepts

| Topic | Recommended Resource |
|---|---|
| Mel spectrograms & log-mel filterbanks | [Filter Banks and Log-Mel Spectrograms](https://haroldbenoit.com/notes/ml/llms/multi-modality/tokenization/audio/filter-banks-and-log-mel-spectrograms) |
| Why mel spectrograms work for audio ML | [Audio Deep Learning Made Simple — Why Mel Spectrograms Perform Better](https://ketanhdoshi.github.io/Audio-Mel/) |
| MFCCs and cepstral analysis | [Speech Processing Book — Mel-Cepstrum and MFCCs](https://speechprocessingbook.aalto.fi/Representations/Melcepstrum.html) |
| LSTM internals | [Understanding LSTMs — colah's blog](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) |
| Attention mechanism | [The Illustrated Transformer — Jay Alammar](https://jalammar.github.io/illustrated-transformer/) |
| BPE tokenization | [Byte Pair Encoding — Hugging Face NLP Course](https://huggingface.co/learn/nlp-course/en/chapter6/5) |

---

### Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│  audio-ml/asr                                            │
│                                                         │
│  SpeechRecognizer (BaseApplication)                     │
│    ├── FeaturePipeline (adapts MelSpectrogramAnalyzer)  │
│    ├── FastConformerEncoder (17-24 Conformer blocks)    │
│    ├── TransducerDecoder (abstract)                     │
│    │   ├── RNNTDecoder (standard transducer)            │
│    │   └── TDTDecoder  (token-and-duration transducer)  │
│    ├── PredictionNetwork (LSTM, shared by both)         │
│    ├── JointNetwork (RNNT or TDT variant)               │
│    ├── CacheManager (streaming state)                   │
│    └── Tokenizer (SentencePiece detokenizer)            │
│                                                         │
│  ComputeBackend interface                               │
│    └── TfjsBackend (wraps tf.* ops)                     │
│                                                         │
├─────────────────────────────────────────────────────────┤
│  @tensorflow/tfjs  (tensor math, GPU dispatch)          │
│    ├── tfjs-backend-wasm    (CPU — XNNPACK, fast)       │
│    ├── tfjs-backend-webgpu  (GPU — WebGPU shaders)      │
│    └── tfjs-backend-webgl   (GPU — WebGL fallback)      │
└─────────────────────────────────────────────────────────┘
```

### RNNT vs TDT — Key Differences

```
RNNT Joint Network:
  encoder[t] + prediction[u] → Linear → ReLU → Linear → [vocab_size]
  Decode: step through every encoder frame one at a time

TDT Joint Network:
  encoder[t] + prediction[u] → Linear → ReLU → Linear → [vocab_size + num_durations]
  Decode: emit token + predicted duration, skip ahead by duration frames
  Result: 2-5x faster inference than RNNT (fewer decoder iterations)
```

The encoder, prediction network, feature pipeline, streaming cache, and tokenizer are all identical. Only the joint network output layer and the greedy decode loop differ.

### File Structure

```
src/
├── analysis/                    # (existing, untouched)
├── applications/                # (existing)
│   └── speech/
│       ├── VAD.ts               # (existing)
│       └── SpeechRecognizer.ts  # NEW — main ASR application
│
├── asr/                         # NEW — all ASR internals
│   ├── index.ts                 # Public exports
│   │
│   ├── compute/                 # Backend abstraction
│   │   ├── Backend.ts           # Interface definition
│   │   ├── TfjsBackend.ts      # TensorFlow.js implementation
│   │   └── types.ts             # TensorHandle, Shape, Dtype
│   │
│   ├── model/                   # Model loading
│   │   ├── SafeTensorsLoader.ts # Parse SafeTensors → tf.tensor
│   │   ├── ModelConfig.ts       # Parse model_config.json
│   │   └── WeightMapper.ts      # NeMo param names → module tree
│   │
│   ├── features/                # Audio preprocessing
│   │   ├── FeaturePipeline.ts   # 80-band log-mel, windowing, normalization
│   │   └── Resampler.ts         # 44.1/48kHz → 16kHz
│   │
│   ├── encoder/                 # FastConformer encoder
│   │   ├── FastConformerEncoder.ts
│   │   ├── ConvSubsampling.ts
│   │   ├── ConformerBlock.ts
│   │   ├── MultiHeadAttention.ts
│   │   ├── ConvModule.ts
│   │   ├── FeedForward.ts
│   │   └── RelativePositionalEncoding.ts
│   │
│   ├── decoder/                 # Transducer decoders (RNNT + TDT)
│   │   ├── PredictionNetwork.ts # LSTM (shared by both RNNT and TDT)
│   │   ├── JointNetwork.ts      # Base joint network
│   │   ├── RNNTJointNetwork.ts  # RNNT variant: outputs [vocab_size]
│   │   ├── TDTJointNetwork.ts   # TDT variant: outputs [vocab_size + num_durations]
│   │   ├── TransducerDecoder.ts # Abstract base with shared decode infrastructure
│   │   ├── RNNTGreedyDecoder.ts # RNNT greedy decode loop (step every frame)
│   │   └── TDTGreedyDecoder.ts  # TDT greedy decode loop (skip by duration)
│   │
│   ├── streaming/               # Cache-aware streaming
│   │   ├── CacheManager.ts      # Attention KV + conv state caches
│   │   ├── ChunkedInference.ts  # Process audio in streaming chunks
│   │   └── Endpointer.ts        # VAD integration for utterance boundaries
│   │
│   └── text/                    # Text processing
│       └── SentencePieceDecoder.ts  # Token IDs → text
│
├── demo/                        # (existing)
│   └── pages/
│       └── SpeechRecognizerDemo.ts  # NEW
│
└── tools/                       # Python utilities (not shipped in npm)
    ├── export_nemo_to_safetensors.py
    ├── export_golden_references.py
    └── export_vocab.py
```

---

## Phase 0: Groundwork (Week 1)

### 0.1 — Python Reference Extraction

Write three Python scripts in `tools/`:

**`export_nemo_to_safetensors.py`** — Converts NeMo checkpoint to SafeTensors:

```python
import nemo.collections.asr as nemo_asr
from safetensors.torch import save_file
import json

model = nemo_asr.models.ASRModel.from_pretrained(
    "nvidia/parakeet_realtime_eou_120m-v1"
    # Other models to export:
    # "nvidia/stt_en_fastconformer_tdt_large"       # English TDT
    # "nvidia/nemotron-speech-streaming-en-0.6b"     # English RNNT streaming
    # "nvidia/parakeet-tdt-0.6b-v3"                  # Multilingual TDT (French + 24 langs)
)

# Export weights
state_dict = {k: v.contiguous() for k, v in model.state_dict().items()}
save_file(state_dict, "parakeet_120m.safetensors")

# Export config — works for both RNNT and TDT models
decoder_type = "tdt" if hasattr(model.cfg, "tdt") or hasattr(model.joint, "duration_head") else "rnnt"
config = {
    "encoder_layers": model.cfg.encoder.n_layers,         # 17
    "d_model": model.cfg.encoder.d_model,                 # 512
    "num_heads": model.cfg.encoder.n_heads,                # 8
    "conv_kernel_size": model.cfg.encoder.conv_kernel_size, # 9
    "ff_expansion_factor": model.cfg.encoder.ff_expansion_factor,
    "subsampling_factor": model.cfg.encoder.subsampling_factor,  # 8
    "vocab_size": model.decoder.prediction.vocab_size,
    "pred_hidden": model.decoder.prediction.hidden_size,
    "num_mel_bands": model.cfg.preprocessor.features,      # 80
    "sample_rate": model.cfg.preprocessor.sample_rate,      # 16000
    "window_size_ms": model.cfg.preprocessor.window_size * 1000,  # 25
    "hop_size_ms": model.cfg.preprocessor.hop_length_ms,   # 10 (= 160 samples)
    "att_context_size": model.cfg.encoder.att_context_size, # [70, 1]
    "decoder_type": decoder_type,
}
# TDT-specific config
if decoder_type == "tdt":
    config["tdt_num_durations"] = model.cfg.model_defaults.tdt_durations  # e.g., [0, 1, 2, 3, 4]
json.dump(config, open("model_config.json", "w"), indent=2)
```

**`export_golden_references.py`** — Saves intermediate tensors at every layer boundary for a test WAV file. This is the primary debugging lifeline.

**`export_vocab.py`** — Extracts the SentencePiece vocabulary as a JSON lookup table.

### 0.2 — Study the Architecture

- Read papers #1 and #2 from the reading list: [Conformer](https://arxiv.org/abs/2005.08100) then [FastConformer](https://arxiv.org/abs/2305.05084)
- Read paper #3 ([RNN-T](https://arxiv.org/abs/1211.3711)) and #5 ([TDT](https://arxiv.org/abs/2304.06795)) for the decoder variants
- Read through NeMo's FastConformer source code — specifically [`ConformerEncoder`](https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/asr/modules/conformer_encoder.py), `ConformerLayer`, `MultiHeadAttention`, `ConvolutionModule`
- Study the [FastConformer YAML config](https://github.com/NVIDIA/NeMo/blob/main/examples/asr/conf/fastconformer/fast-conformer_transducer_bpe.yaml) for all hyperparameters
- Trace a forward pass with a debugger, noting shapes at every step
- Document the exact layer sequence, parameter names, and dimension conventions

### 0.3 — Set Up the Project

```bash
# Add dependencies
yarn add @tensorflow/tfjs @tensorflow/tfjs-backend-wasm
# Dev dependencies for testing
yarn add -D @tensorflow/tfjs-backend-cpu vitest
```

Add new exports to `package.json`:

```json
"exports": {
  ".": "./dist/analysis/index.js",
  "./applications": "./dist/applications/index.js",
  "./asr": "./dist/asr/index.js"
}
```

**Milestone**: SafeTensors file + model config + golden references + SentencePiece vocab, all exported and ready. Project scaffolded with tfjs dependency.

---

## Phase 1: Compute Backend Abstraction (Week 2)

TensorFlow.js docs: [Guide](https://www.tensorflow.org/js/guide) |
[npm](https://www.npmjs.com/package/@tensorflow/tfjs) |
[Tensors & Ops](https://www.tensorflow.org/js/guide/tensors_operations)

### 1.1 — Backend Interface

The key design decision: model code never imports `@tensorflow/tfjs` directly. Everything goes through an abstraction, so tfjs is swappable later.

```typescript
// src/asr/compute/types.ts
export type TensorHandle = unknown;  // opaque — tf.Tensor internally
export type Shape = readonly number[];
export type Dtype = 'float32' | 'float16' | 'int32' | 'int64';
```

```typescript
// src/asr/compute/Backend.ts
export interface ComputeBackend {
  // Tensor creation
  tensor(data: Float32Array | Int32Array, shape: Shape): TensorHandle;
  zeros(shape: Shape): TensorHandle;
  dispose(t: TensorHandle): void;

  // Core ops
  matmul(a: TensorHandle, b: TensorHandle): TensorHandle;
  add(a: TensorHandle, b: TensorHandle): TensorHandle;
  mul(a: TensorHandle, b: TensorHandle): TensorHandle;
  scale(a: TensorHandle, s: number): TensorHandle;

  // Reductions
  softmax(x: TensorHandle, axis: number): TensorHandle;
  layerNorm(x: TensorHandle, weight: TensorHandle, bias: TensorHandle, eps: number): TensorHandle;
  batchNorm(x: TensorHandle, mean: TensorHandle, variance: TensorHandle,
            scale: TensorHandle, offset: TensorHandle, eps: number): TensorHandle;

  // Activations
  relu(x: TensorHandle): TensorHandle;
  silu(x: TensorHandle): TensorHandle;
  gelu(x: TensorHandle): TensorHandle;
  sigmoid(x: TensorHandle): TensorHandle;
  tanh(x: TensorHandle): TensorHandle;

  // Convolutions
  conv1d(input: TensorHandle, kernel: TensorHandle, stride: number, padding: number): TensorHandle;
  conv2d(input: TensorHandle, kernel: TensorHandle, strides: [number, number], padding: string): TensorHandle;
  depthwiseConv1d(input: TensorHandle, kernel: TensorHandle, stride: number, padding: number): TensorHandle;

  // Shape ops
  reshape(x: TensorHandle, shape: Shape): TensorHandle;
  transpose(x: TensorHandle, perm: number[]): TensorHandle;
  slice(x: TensorHandle, begin: number[], size: number[]): TensorHandle;
  concat(tensors: TensorHandle[], axis: number): TensorHandle;
  split(x: TensorHandle, numSplits: number, axis: number): TensorHandle[];
  gather(x: TensorHandle, indices: TensorHandle, axis: number): TensorHandle;

  // Data transfer
  getData(t: TensorHandle): Promise<Float32Array>;
  getShape(t: TensorHandle): Shape;
}
```

### 1.2 — TensorFlow.js Implementation

```typescript
// src/asr/compute/TfjsBackend.ts
import * as tf from '@tensorflow/tfjs';
import type { ComputeBackend, TensorHandle, Shape } from './Backend';

export class TfjsBackend implements ComputeBackend {
  constructor(backend: 'wasm' | 'webgpu' | 'webgl' | 'cpu' = 'wasm') {
    tf.setBackend(backend);
  }

  tensor(data: Float32Array, shape: Shape): TensorHandle {
    return tf.tensor(data, shape as number[]);
  }

  matmul(a: TensorHandle, b: TensorHandle): TensorHandle {
    return tf.matMul(a as tf.Tensor, b as tf.Tensor);
  }

  layerNorm(x: TensorHandle, weight: TensorHandle, bias: TensorHandle, eps: number): TensorHandle {
    return tf.tidy(() => {
      const t = x as tf.Tensor;
      const axis = t.rank - 1;
      const { mean, variance } = tf.moments(t, [axis], true);
      const normalized = tf.div(tf.sub(t, mean), tf.sqrt(tf.add(variance, eps)));
      return tf.add(tf.mul(normalized, weight as tf.Tensor), bias as tf.Tensor);
    });
  }

  silu(x: TensorHandle): TensorHandle {
    return tf.tidy(() => {
      const t = x as tf.Tensor;
      return tf.mul(t, tf.sigmoid(t));
    });
  }

  depthwiseConv1d(input: TensorHandle, kernel: TensorHandle, stride: number, padding: number): TensorHandle {
    return tf.tidy(() => {
      // tfjs has depthwiseConv2d but not 1d — reshape to [B, 1, T, C] and use [1, K] filter
      const inp3d = input as tf.Tensor3D; // [B, T, C]
      const inp4d = inp3d.expandDims(1);  // [B, 1, T, C]
      const kern = (kernel as tf.Tensor).expandDims(0); // [1, K, C_in, multiplier]
      const result = tf.depthwiseConv2d(inp4d as tf.Tensor4D, kern as tf.Tensor4D,
        [1, stride], padding === 0 ? 'valid' : 'same');
      return result.squeeze([1]); // back to [B, T, C]
    });
  }

  dispose(t: TensorHandle): void {
    (t as tf.Tensor).dispose();
  }

  async getData(t: TensorHandle): Promise<Float32Array> {
    return (t as tf.Tensor).data() as Promise<Float32Array>;
  }

  // ... remaining ops follow the same pattern
}
```

### 1.3 — Memory Management Helper

tfjs leaks memory if tensors aren't disposed. A scope helper addresses this:

```typescript
export class ComputeScope {
  private tensors: TensorHandle[] = [];

  track<T extends TensorHandle>(t: T): T {
    this.tensors.push(t);
    return t;
  }

  keep(t: TensorHandle): void {
    const idx = this.tensors.indexOf(t);
    if (idx >= 0) this.tensors.splice(idx, 1);
  }

  dispose(backend: ComputeBackend): void {
    for (const t of this.tensors) backend.dispose(t);
    this.tensors = [];
  }
}
```

Each model module uses scopes to clean up intermediate tensors while keeping outputs:

```typescript
forward(x: TensorHandle): TensorHandle {
  const scope = new ComputeScope();
  const a = scope.track(this.backend.matmul(x, this.weight));
  const b = scope.track(this.backend.add(a, this.bias));
  const out = this.backend.silu(b);
  scope.dispose(this.backend);  // disposes a and b, not out
  return out;
}
```

**Milestone**: `ComputeBackend` interface defined, `TfjsBackend` implemented with all ~25 ops, memory scoping works. Unit tests confirm ops produce correct outputs.

---

## Phase 2: Weight Loading (Week 2, parallel with Phase 1)

### 2.1 — SafeTensors Parser

Format spec: [SafeTensors docs](https://huggingface.co/docs/safetensors) — 8-byte header length, JSON header, then raw tensor data.

```typescript
// src/asr/model/SafeTensorsLoader.ts
export async function loadSafeTensors(
  source: string | ArrayBuffer,
  backend: ComputeBackend
): Promise<Map<string, TensorHandle>> {
  const buffer = typeof source === 'string'
    ? await (await fetch(source)).arrayBuffer()
    : source;

  const headerLen = Number(new DataView(buffer).getBigUint64(0, true));
  const headerJson = new TextDecoder().decode(new Uint8Array(buffer, 8, headerLen));
  const header: Record<string, { dtype: string; shape: number[]; data_offsets: [number, number] }>
    = JSON.parse(headerJson);

  const dataStart = 8 + headerLen;
  const tensors = new Map<string, TensorHandle>();

  for (const [name, meta] of Object.entries(header)) {
    if (name === '__metadata__') continue;
    const [start, end] = meta.data_offsets;
    const byteLength = end - start;
    const data = new Float32Array(buffer.slice(dataStart + start, dataStart + start + byteLength));
    tensors.set(name, backend.tensor(data, meta.shape));
  }

  return tensors;
}
```

### 2.2 — Model Config Parser

```typescript
// src/asr/model/ModelConfig.ts
export type DecoderType = 'rnnt' | 'tdt';

export interface FastConformerConfig {
  encoderLayers: number;      // 17 (Parakeet) or 24 (Nemotron)
  dModel: number;             // 512
  numHeads: number;           // 8
  convKernelSize: number;     // 9
  ffExpansionFactor: number;  // 4
  subsamplingFactor: number;  // 8
  vocabSize: number;
  predHidden: number;
  numMelBands: number;        // 80
  sampleRate: number;         // 16000
  windowSizeMs: number;       // 25
  hopSizeMs: number;          // 10
  attContextSize: [number, number]; // [70, 1]
  decoderType: DecoderType;   // 'rnnt' or 'tdt'
  tdtNumDurations?: number;   // TDT only: number of duration bins (e.g., 4 → skip 0-3 frames)
}

export function parseModelConfig(json: string): FastConformerConfig {
  const raw = JSON.parse(json);
  return {
    encoderLayers: raw.encoder_layers,
    dModel: raw.d_model,
    decoderType: raw.decoder_type ?? 'rnnt',
    tdtNumDurations: raw.tdt_num_durations,
    // ... direct mapping
  };
}
```

### 2.3 — Weight Mapper

Maps NeMo parameter names to the module hierarchy. Validates that every weight is consumed:

```typescript
// src/asr/model/WeightMapper.ts
export function mapWeights(
  weights: Map<string, TensorHandle>,
  config: FastConformerConfig
): ModelWeights {
  const consumed = new Set<string>();

  function get(name: string): TensorHandle {
    if (!weights.has(name)) throw new Error(`Missing weight: ${name}`);
    consumed.add(name);
    return weights.get(name)!;
  }

  const encoder = {
    subsampling: {
      conv1_weight: get('encoder.pre_encode.conv.0.weight'),
      conv1_bias: get('encoder.pre_encode.conv.0.bias'),
      conv2_weight: get('encoder.pre_encode.conv.2.weight'),
      conv2_bias: get('encoder.pre_encode.conv.2.bias'),
      out_weight: get('encoder.pre_encode.out.0.weight'),
      out_bias: get('encoder.pre_encode.out.0.bias'),
    },
    layers: Array.from({ length: config.encoderLayers }, (_, i) => ({
      ffn1: {
        w1: get(`encoder.layers.${i}.fc1.weight`),
        b1: get(`encoder.layers.${i}.fc1.bias`),
        w2: get(`encoder.layers.${i}.fc2.weight`),
        b2: get(`encoder.layers.${i}.fc2.bias`),
        norm: {
          weight: get(`encoder.layers.${i}.norm_feed_forward1.weight`),
          bias: get(`encoder.layers.${i}.norm_feed_forward1.bias`),
        },
      },
      attn: {
        // Q, K, V, output projections + relative pos encoding weights
        // ... map all attention weights
      },
      conv: {
        // pointwise1, depthwise, batchnorm, pointwise2 weights
        // ... map all conv module weights
      },
      ffn2: {
        // same structure as ffn1
      },
      finalNorm: {
        weight: get(`encoder.layers.${i}.norm_out.weight`),
        bias: get(`encoder.layers.${i}.norm_out.bias`),
      },
    })),
  };

  // Validate all weights consumed
  const unconsumed = [...weights.keys()].filter(k => !consumed.has(k));
  if (unconsumed.length > 0) {
    console.warn('Unconsumed weights:', unconsumed);
  }

  return { encoder, decoder: { /* prediction + joint weights */ } };
}
```

**Note**: The exact NeMo parameter names will need to be discovered from the checkpoint. Run `model.state_dict().keys()` in Python and print them all — this is the map to build from.

**Milestone**: Can load `parakeet_120m.safetensors` + `model_config.json` in Node.js, get a typed `ModelWeights` object with all parameters as tf.Tensors.

---

## Phase 3: Feature Pipeline (Week 3)

Background reading: [Filter Banks and Log-Mel Spectrograms](https://haroldbenoit.com/notes/ml/llms/multi-modality/tokenization/audio/filter-banks-and-log-mel-spectrograms)
and [Why Mel Spectrograms Perform Better](https://ketanhdoshi.github.io/Audio-Mel/).

### 3.1 — ASR Feature Extractor

Adapts the existing `MelSpectrogramAnalyzer` to match NeMo's exact preprocessing. The critical parameters for Parakeet:

```typescript
// src/asr/features/FeaturePipeline.ts
export class FeaturePipeline {
  private melAnalyzer: MelSpectrogramAnalyzer;
  private windowSize: number;   // 400 samples (25ms at 16kHz)
  private hopSize: number;      // 160 samples (10ms at 16kHz)
  private buffer: Float32Array;
  private bufferPos: number = 0;

  constructor(config: FastConformerConfig) {
    this.windowSize = Math.round(config.sampleRate * config.windowSizeMs / 1000);
    this.hopSize = Math.round(config.sampleRate * config.hopSizeMs / 1000);

    this.melAnalyzer = new MelSpectrogramAnalyzer({
      sampleRate: config.sampleRate,
      fftSize: 512,          // next power of 2 above windowSize
      melBands: config.numMelBands,  // 80
    });

    this.buffer = new Float32Array(this.windowSize);
  }

  extractFeatures(audio: Float32Array): TensorHandle {
    // 1. Apply pre-emphasis filter (y[n] = x[n] - 0.97 * x[n-1])
    // 2. Frame into overlapping windows (25ms window, 10ms hop)
    // 3. Apply Hann window to each frame
    // 4. Compute mel spectrogram for each frame
    // 5. Log scale
    // 6. Normalize (subtract mean, divide by std)
    // 7. Return as [1, numFrames, 80] tensor
  }
}
```

**Key differences from the existing `MelSpectrogramAnalyzer`**:

1. **Power spectrum**: Square the magnitudes before mel filtering (`mag * mag`, not just `mag`). NeMo uses power spectrum.
2. **Hann window**: Apply a Hann window to each frame before FFT. The current analyzer doesn't window.
3. **Windowing/hopping**: The current analyzer takes one frame at a time. The feature pipeline needs to handle the overlapping window/hop internally.
4. **Normalization**: Per-feature mean/variance normalization.

### 3.2 — Resampler

For mic input at 44.1/48kHz:

```typescript
// src/asr/features/Resampler.ts
export class Resampler {
  private fromRate: number;
  private toRate: number;

  constructor(fromRate: number, toRate: number = 16000) {
    this.fromRate = fromRate;
    this.toRate = toRate;
  }

  resample(input: Float32Array): Float32Array {
    if (this.fromRate === this.toRate) return input;
    // Linear interpolation for simplicity (good enough for speech)
    // Upgrade to sinc interpolation later if quality matters
    const ratio = this.toRate / this.fromRate;
    const outputLen = Math.round(input.length * ratio);
    const output = new Float32Array(outputLen);
    for (let i = 0; i < outputLen; i++) {
      const srcIdx = i / ratio;
      const lo = Math.floor(srcIdx);
      const hi = Math.min(lo + 1, input.length - 1);
      const frac = srcIdx - lo;
      output[i] = input[lo] * (1 - frac) + input[hi] * frac;
    }
    return output;
  }
}
```

### 3.3 — Validate Against NeMo

Feed the same WAV to NeMo's preprocessor and the `FeaturePipeline`. Compare the output `[1, T, 80]` tensor element-by-element. Must match within `1e-4`. Any mismatch here cascades through 17 encoder layers and ruins everything downstream.

**Milestone**: Feature pipeline produces identical mel features to NeMo for test audio.

---

## Phase 4: FastConformer Encoder (Weeks 4–6)

This is the core of the project. Build each module, validate layer by layer.

### 4.1 — Linear Layer Helper

Used everywhere — wrap it once:

```typescript
// Common building block
class Linear {
  constructor(
    private backend: ComputeBackend,
    private weight: TensorHandle,  // [out, in]
    private bias: TensorHandle | null,
  ) {}

  forward(x: TensorHandle): TensorHandle {
    // x: [B, T, in] → [B, T, out]
    const wT = this.backend.transpose(this.weight, [1, 0]);
    let out = this.backend.matmul(x, wT);
    if (this.bias) out = this.backend.add(out, this.bias);
    this.backend.dispose(wT);
    return out;
  }
}
```

### 4.2 — Conv Subsampling (Week 4, days 1–2)

Two Conv2D layers that downsample time by 8x:

```typescript
// src/asr/encoder/ConvSubsampling.ts
class ConvSubsampling {
  forward(melFeatures: TensorHandle): TensorHandle {
    // Input: [B, T, 80]
    // Reshape to [B, T, 80, 1] for conv2d
    // Conv2D(1→channels, kernel=3x3, stride=2x2) → ReLU
    // Conv2D(channels→channels, kernel=3x3, stride=2x2) → ReLU
    // This gives 4x downsampling from 2 stride-2 convs
    // (FastConformer gets 8x via an additional pooling or larger stride)
    // Reshape [B, T/4, channels*features] → Linear → [B, T/4, d_model]
    // Output: [B, T/8, d_model]
  }
}
```

Validate: compare output shape and values against NeMo's `encoder.pre_encode()`.

### 4.3 — Feed-Forward Module (Week 4, days 3–4)

```typescript
// src/asr/encoder/FeedForward.ts
class FeedForward {
  private norm: LayerNormParams;
  private linear1: Linear;   // [d_model, d_model * expansion_factor]
  private linear2: Linear;   // [d_model * expansion_factor, d_model]

  forward(x: TensorHandle): TensorHandle {
    const scope = new ComputeScope();
    const normed = scope.track(this.backend.layerNorm(x, this.norm.weight, this.norm.bias, 1e-5));
    const hidden = scope.track(this.linear1.forward(normed));
    const activated = scope.track(this.backend.silu(hidden));
    const out = this.linear2.forward(activated);
    scope.dispose(this.backend);
    return out;
  }
}
```

### 4.4 — Multi-Head Self-Attention (Week 4–5, ~4 days)

The hardest single module. FastConformer uses **relative positional encoding**
(see [paper #7](https://www.isca-archive.org/interspeech_2020/pham20_interspeech.html) and
[The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) for background):

```typescript
// src/asr/encoder/MultiHeadAttention.ts
class MultiHeadAttention {
  private numHeads: number;
  private headDim: number;
  private qProj: Linear;
  private kProj: Linear;
  private vProj: Linear;
  private outProj: Linear;
  private posEncoding: RelativePositionalEncoding;

  forward(x: TensorHandle, mask?: TensorHandle): TensorHandle {
    const B = this.backend.getShape(x)[0];
    const T = this.backend.getShape(x)[1];

    // Project to Q, K, V
    const q = this.splitHeads(this.qProj.forward(x));  // [B, heads, T, d_head]
    const k = this.splitHeads(this.kProj.forward(x));
    const v = this.splitHeads(this.vProj.forward(x));

    // Scaled dot-product attention with relative position bias
    const scale = 1.0 / Math.sqrt(this.headDim);
    const scores = this.backend.scale(this.backend.matmul(q, this.backend.transpose(k, [0, 1, 3, 2])), scale);

    // Add relative positional encoding bias
    const posScores = this.posEncoding.forward(q, T);
    const totalScores = this.backend.add(scores, posScores);

    // Optional attention mask (for cache-aware streaming)
    // Apply mask, softmax, matmul with V
    const attnWeights = this.backend.softmax(totalScores, -1);
    const attnOut = this.backend.matmul(attnWeights, v);

    // Merge heads and project
    const merged = this.mergeHeads(attnOut);  // [B, T, d_model]
    return this.outProj.forward(merged);
  }

  private splitHeads(x: TensorHandle): TensorHandle {
    // [B, T, d_model] → [B, T, heads, d_head] → [B, heads, T, d_head]
  }

  private mergeHeads(x: TensorHandle): TensorHandle {
    // [B, heads, T, d_head] → [B, T, heads, d_head] → [B, T, d_model]
  }
}
```

**Validate carefully**: Print attention weights for a single head on the test input. Compare to NeMo's attention weights. This catches off-by-one errors in positional encoding, wrong transpose orders, and scaling issues.

### 4.5 — Convolution Module (Week 5, ~3 days)

```typescript
// src/asr/encoder/ConvModule.ts
class ConvModule {
  forward(x: TensorHandle): TensorHandle {
    // LayerNorm
    let h = this.backend.layerNorm(x, this.norm.weight, this.norm.bias, 1e-5);

    // Pointwise conv (expand: d_model → 2*d_model)
    h = this.pointwiseConv1.forward(h);

    // GLU gating: split in half, sigmoid gate
    const [a, b] = this.backend.split(h, 2, -1);
    h = this.backend.mul(a, this.backend.sigmoid(b));

    // Depthwise conv1d (kernel_size=9, groups=d_model)
    h = this.backend.depthwiseConv1d(h, this.depthwiseWeight, 1, this.padding);

    // BatchNorm
    h = this.backend.batchNorm(h, this.bnMean, this.bnVar, this.bnScale, this.bnOffset, 1e-5);

    // SiLU activation
    h = this.backend.silu(h);

    // Pointwise conv (project: d_model → d_model)
    h = this.pointwiseConv2.forward(h);

    return h;
  }
}
```

### 4.6 — Full Conformer Block (Week 5)

```typescript
// src/asr/encoder/ConformerBlock.ts
class ConformerBlock {
  forward(x: TensorHandle): TensorHandle {
    // Macaron sandwich
    x = this.backend.add(x, this.backend.scale(this.ffn1.forward(x), 0.5));
    x = this.backend.add(x, this.attn.forward(x));
    x = this.backend.add(x, this.conv.forward(x));
    x = this.backend.add(x, this.backend.scale(this.ffn2.forward(x), 0.5));
    x = this.backend.layerNorm(x, this.finalNorm.weight, this.finalNorm.bias, 1e-5);
    return x;
  }
}
```

### 4.7 — Full Encoder (Week 6)

```typescript
// src/asr/encoder/FastConformerEncoder.ts
export class FastConformerEncoder {
  private subsampling: ConvSubsampling;
  private blocks: ConformerBlock[];

  forward(melFeatures: TensorHandle): TensorHandle {
    let x = this.subsampling.forward(melFeatures);
    for (const block of this.blocks) {
      x = block.forward(x);
    }
    return x;
  }
}
```

### 4.8 — End-to-End Encoder Validation

Feed the golden mel features from Phase 0 through the full encoder. Compare final output `[1, T/8, 512]` to NeMo's encoder output. Must match within `1e-3`. If it doesn't, use the per-layer golden references to binary-search which layer is wrong.

**Milestone**: Encoder forward pass matches NeMo. This is the big one — once this works, the rest is straightforward.

---

## Phase 5: Transducer Decoders — RNNT + TDT (Weeks 7–8)

The encoder is done. Now build both decoder variants. They share the prediction network
and differ only in the joint network output layer and the decode loop.

### 5.1 — Prediction Network (LSTM) — Shared

Identical for both RNNT and TDT. The prediction network conditions on previously
emitted tokens, regardless of how frame advancement works.
(See [Understanding LSTMs](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) for LSTM internals.)

```typescript
// src/asr/decoder/PredictionNetwork.ts
export class PredictionNetwork {
  private embedding: TensorHandle;  // [vocab_size, embed_dim]
  private lstmWeightIH: TensorHandle; // [4*hidden, embed_dim]
  private lstmWeightHH: TensorHandle; // [4*hidden, hidden]
  private lstmBiasIH: TensorHandle;
  private lstmBiasHH: TensorHandle;
  private outputProj: Linear;

  step(tokenId: number, h: TensorHandle, c: TensorHandle):
    { output: TensorHandle; h: TensorHandle; c: TensorHandle } {
    // 1. Embed the token
    const emb = this.backend.gather(this.embedding, /* token index */);

    // 2. LSTM cell
    //    gates = emb * W_ih + h * W_hh + b_ih + b_hh
    //    i, f, g, o = split(gates, 4)
    //    c_new = sigmoid(f) * c + sigmoid(i) * tanh(g)
    //    h_new = sigmoid(o) * tanh(c_new)

    // 3. Output projection
    const output = this.outputProj.forward(hNew);

    return { output, h: hNew, c: cNew };
  }

  initialState(): { h: TensorHandle; c: TensorHandle } {
    return {
      h: this.backend.zeros([1, this.hiddenSize]),
      c: this.backend.zeros([1, this.hiddenSize]),
    };
  }
}
```

### 5.2 — Joint Networks (RNNT and TDT Variants)

The joint network combines encoder and prediction outputs. The only structural
difference: TDT has an additional duration head.

```typescript
// src/asr/decoder/JointNetwork.ts — Base with shared projection logic
export class JointNetwork {
  protected encoderProj: Linear;
  protected predictionProj: Linear;

  protected computeJoint(encoderFrame: TensorHandle, predictionOut: TensorHandle): TensorHandle {
    const enc = this.encoderProj.forward(encoderFrame);
    const pred = this.predictionProj.forward(predictionOut);
    return this.backend.relu(this.backend.add(enc, pred));
  }
}
```

```typescript
// src/asr/decoder/RNNTJointNetwork.ts
export class RNNTJointNetwork extends JointNetwork {
  private outputProj: Linear;  // → [vocab_size]

  forward(encoderFrame: TensorHandle, predictionOut: TensorHandle): { tokenLogits: TensorHandle } {
    const joint = this.computeJoint(encoderFrame, predictionOut);
    const tokenLogits = this.outputProj.forward(joint);
    this.backend.dispose(joint);
    return { tokenLogits };  // [vocab_size]
  }
}
```

```typescript
// src/asr/decoder/TDTJointNetwork.ts
export class TDTJointNetwork extends JointNetwork {
  private tokenProj: Linear;     // → [vocab_size]
  private durationProj: Linear;  // → [num_durations]

  forward(encoderFrame: TensorHandle, predictionOut: TensorHandle):
    { tokenLogits: TensorHandle; durationLogits: TensorHandle } {
    const joint = this.computeJoint(encoderFrame, predictionOut);
    const tokenLogits = this.tokenProj.forward(joint);
    const durationLogits = this.durationProj.forward(joint);
    this.backend.dispose(joint);
    return { tokenLogits, durationLogits };  // [vocab_size], [num_durations]
  }
}
```

### 5.3 — RNNT Greedy Decode

Standard transducer ([paper #3](https://arxiv.org/abs/1211.3711)): iterate every encoder frame, emit tokens until blank.

```typescript
// src/asr/decoder/RNNTGreedyDecoder.ts
export class RNNTGreedyDecoder {
  private predNet: PredictionNetwork;
  private jointNet: RNNTJointNetwork;
  private blankId: number = 0;
  private maxSymbolsPerStep: number = 10;

  async decode(encoderOutput: TensorHandle): Promise<number[]> {
    const T = this.backend.getShape(encoderOutput)[1];
    const tokens: number[] = [];
    let { h, c } = this.predNet.initialState();
    let lastToken = this.blankId;

    for (let t = 0; t < T; t++) {
      const encFrame = this.backend.slice(encoderOutput, [0, t, 0], [1, 1, -1]);
      let symbolsEmitted = 0;

      while (symbolsEmitted < this.maxSymbolsPerStep) {
        const { output: predOut, h: hNew, c: cNew } = this.predNet.step(lastToken, h, c);
        const { tokenLogits } = this.jointNet.forward(encFrame, predOut);
        const logitsData = await this.backend.getData(tokenLogits);
        const token = argmax(logitsData);

        this.backend.dispose(tokenLogits);
        this.backend.dispose(predOut);

        if (token === this.blankId) {
          this.backend.dispose(hNew);
          this.backend.dispose(cNew);
          break;
        }

        tokens.push(token);
        this.backend.dispose(h);
        this.backend.dispose(c);
        h = hNew;
        c = cNew;
        lastToken = token;
        symbolsEmitted++;
      }
    }

    this.backend.dispose(h);
    this.backend.dispose(c);
    return tokens;
  }
}
```

### 5.4 — TDT Greedy Decode

Token-and-Duration Transducer ([paper #5](https://arxiv.org/abs/2304.06795)): the joint network
predicts both a token and a duration (how many encoder frames to skip). This makes
decoding 2-5x faster than RNNT because the decoder doesn't iterate every frame.

```typescript
// src/asr/decoder/TDTGreedyDecoder.ts
export class TDTGreedyDecoder {
  private predNet: PredictionNetwork;
  private jointNet: TDTJointNetwork;
  private blankId: number = 0;
  private durations: number[];    // e.g., [0, 1, 2, 3, 4] — the duration vocab
  private maxSymbolsPerStep: number = 10;

  constructor(config: FastConformerConfig, ...) {
    // Duration bins from model config — maps duration index to actual frame skip count
    this.durations = config.tdtDurations ?? [0, 1, 2, 3, 4];
  }

  async decode(encoderOutput: TensorHandle): Promise<number[]> {
    const T = this.backend.getShape(encoderOutput)[1];
    const tokens: number[] = [];
    let { h, c } = this.predNet.initialState();
    let lastToken = this.blankId;
    let t = 0;

    while (t < T) {
      const encFrame = this.backend.slice(encoderOutput, [0, t, 0], [1, 1, -1]);
      let symbolsEmitted = 0;

      while (symbolsEmitted < this.maxSymbolsPerStep) {
        const { output: predOut, h: hNew, c: cNew } = this.predNet.step(lastToken, h, c);
        const { tokenLogits, durationLogits } = this.jointNet.forward(encFrame, predOut);

        const tokenData = await this.backend.getData(tokenLogits);
        const durData = await this.backend.getData(durationLogits);
        const token = argmax(tokenData);
        const durIdx = argmax(durData);
        const duration = this.durations[durIdx];

        this.backend.dispose(tokenLogits);
        this.backend.dispose(durationLogits);
        this.backend.dispose(predOut);

        if (token === this.blankId) {
          this.backend.dispose(hNew);
          this.backend.dispose(cNew);
          // TDT: even on blank, advance by predicted duration
          t += Math.max(1, duration);
          break;
        }

        tokens.push(token);
        this.backend.dispose(h);
        this.backend.dispose(c);
        h = hNew;
        c = cNew;
        lastToken = token;
        symbolsEmitted++;

        // TDT: advance encoder position by predicted duration after emitting token
        t += duration;
        if (duration > 0) break;  // moved forward — need new encoder frame
      }
    }

    this.backend.dispose(h);
    this.backend.dispose(c);
    return tokens;
  }
}
```

**Key difference from RNNT**: In RNNT, the outer loop is `for (let t = 0; t < T; t++)` —
it visits every frame. In TDT, `t` advances by the predicted duration, skipping frames.
On typical speech, the model learns to skip 2-4 frames at a time during silence/sustained
sounds, cutting decoder iterations by 2-5x.

### 5.5 — Decoder Factory

Auto-select the right decoder based on model config:

```typescript
// src/asr/decoder/createDecoder.ts
export function createDecoder(
  config: FastConformerConfig,
  backend: ComputeBackend,
  weights: DecoderWeights,
): RNNTGreedyDecoder | TDTGreedyDecoder {
  const predNet = new PredictionNetwork(backend, weights.prediction);

  if (config.decoderType === 'tdt') {
    const jointNet = new TDTJointNetwork(backend, weights.joint);
    return new TDTGreedyDecoder(predNet, jointNet, config);
  }

  const jointNet = new RNNTJointNetwork(backend, weights.joint);
  return new RNNTGreedyDecoder(predNet, jointNet, config);
}
```

**Important note on performance**: Both decode loops call the prediction network and joint
network **per encoder frame, potentially multiple times**. Each call is a small matmul.
On GPU, these tiny ops have high dispatch overhead. For the initial version, running
the decode loop on CPU (`await getData()` to pull encoder output to CPU, then do the
loop with typed arrays) may actually be faster than dispatching hundreds of tiny GPU
kernels. Profile this and decide. TDT partially mitigates this — fewer iterations means
fewer tiny GPU dispatches.

### 5.6 — SentencePiece Detokenizer

See [paper #8](https://arxiv.org/abs/1808.06226) and [google/sentencepiece](https://github.com/google/sentencepiece) for the format.

```typescript
// src/asr/text/SentencePieceDecoder.ts
export class SentencePieceDecoder {
  private vocab: string[];

  constructor(vocabJson: string) {
    this.vocab = JSON.parse(vocabJson); // ["<blank>", "▁the", "▁a", "s", "e", ...]
  }

  decode(tokenIds: number[]): string {
    return tokenIds
      .map(id => this.vocab[id] ?? '')
      .join('')
      .replace(/▁/g, ' ')
      .trim();
  }
}
```

### 5.7 — End-to-End Validation

WAV → mel features → encoder → decoder → token IDs → text. Compare against NeMo's
transcription for multiple test files. The text should be **identical** (not just close —
greedy decoding is deterministic).

Test with both an RNNT model (Parakeet 120M) and a TDT model if available. The encoder
output should be identical — only the decoder path differs.

**Milestone**: Full offline transcription works for both RNNT and TDT models. Feed a WAV file, get correct text out.

---

## Phase 6: Integration (Week 9)

### 6.1 — SpeechRecognizer Application

Following the `BaseApplication` pattern:

```typescript
// src/applications/speech/SpeechRecognizer.ts
export interface SpeechRecognizerConfig extends ApplicationConfig {
  modelPath: string;
  configPath: string;
  vocabPath: string;
  backend?: 'wasm' | 'webgpu' | 'webgl' | 'cpu';
}

export interface ASRResult {
  text: string;
  isFinal: boolean;
  latencyMs: number;
  decoderType: 'rnnt' | 'tdt';  // which decoder variant was used
}

export class SpeechRecognizer extends BaseApplication {
  private featurePipeline: FeaturePipeline;
  private encoder: FastConformerEncoder;
  private decoder: RNNTGreedyDecoder | TDTGreedyDecoder;  // auto-selected from config
  private tokenizer: SentencePieceDecoder;
  private audioBuffer: Float32Array[] = [];
  private isLoaded = false;

  async load(): Promise<void> {
    const backend = new TfjsBackend(this.backendType);
    const config = parseModelConfig(await fetchText(this.configPath));
    const weights = await loadSafeTensors(this.modelPath, backend);
    const modelWeights = mapWeights(weights, config);

    this.featurePipeline = new FeaturePipeline(config);
    this.encoder = new FastConformerEncoder(backend, modelWeights.encoder, config);
    this.decoder = createDecoder(config, backend, modelWeights.decoder);  // RNNT or TDT
    this.tokenizer = new SentencePieceDecoder(await fetchText(this.vocabPath));

    this.isLoaded = true;
    this.emit('ready', { decoderType: config.decoderType });
  }

  processFrame(pcm: Float32Array): void {
    this.audioBuffer.push(pcm);
    // Accumulate until we have enough for recognition
    // (details depend on streaming vs. offline mode)
  }

  async transcribe(audio: Float32Array): Promise<ASRResult> {
    const start = performance.now();
    const mel = this.featurePipeline.extractFeatures(audio);
    const encoded = this.encoder.forward(mel);
    const tokenIds = await this.decoder.decode(encoded);
    const text = this.tokenizer.decode(tokenIds);
    this.backend.dispose(mel);
    this.backend.dispose(encoded);

    return {
      text,
      isFinal: true,
      latencyMs: performance.now() - start,
    };
  }

  reset(): void {
    super.reset();
    this.audioBuffer = [];
  }
}
```

### 6.2 — Package Export

```typescript
// src/asr/index.ts
export { SpeechRecognizer, type SpeechRecognizerConfig, type ASRResult }
  from '../applications/speech/SpeechRecognizer';
export { FastConformerEncoder } from './encoder/FastConformerEncoder';
export { RNNTGreedyDecoder } from './decoder/RNNTGreedyDecoder';
export { TDTGreedyDecoder } from './decoder/TDTGreedyDecoder';
export { createDecoder } from './decoder/createDecoder';
export { FeaturePipeline } from './features/FeaturePipeline';
export { TfjsBackend } from './compute/TfjsBackend';
export type { ComputeBackend } from './compute/Backend';
export type { DecoderType } from './model/ModelConfig';
```

Usage:

```typescript
import { SpeechRecognizer } from 'audio-ml/asr';

const recognizer = new SpeechRecognizer({
  sampleRate: 16000,
  modelPath: './parakeet_120m.safetensors',
  configPath: './model_config.json',
  vocabPath: './vocab.json',
  backend: 'wasm',
});

await recognizer.load();
const result = await recognizer.transcribe(audioFloat32Array);
console.log(result.text);
```

**Milestone**: Working `npm install audio-ml` → import → load model → transcribe audio → get text. Clean API.

---

## Phase 7: Cache-Aware Streaming (Weeks 10–11)

Based on [paper #6: Stateful Conformer with Cache-based Inference](https://arxiv.org/abs/2312.17279).
Reference implementation: [NeMo cache-aware streaming script](https://github.com/NVIDIA/NeMo/blob/main/examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py).

### 7.1 — Cache Manager

```typescript
// src/asr/streaming/CacheManager.ts
export interface StreamingCache {
  attentionKV: Array<{ k: TensorHandle; v: TensorHandle }>;  // per layer
  convStates: TensorHandle[];   // per layer
  predictionState: { h: TensorHandle; c: TensorHandle };
  lastToken: number;
  encodedFrameCount: number;
  // TDT-specific: tracks the current frame offset within the latest encoder chunk,
  // since TDT may not have consumed all frames before the chunk boundary
  tdtFrameOffset: number;
}

export class CacheManager {
  create(config: FastConformerConfig, backend: ComputeBackend): StreamingCache;
  dispose(cache: StreamingCache, backend: ComputeBackend): void;
}
```

### 7.2 — Streaming Encoder

Each Conformer block's forward method gets a cache-aware variant:

```typescript
// Modified attention forward
forwardStreaming(
  chunk: TensorHandle,           // [B, chunk_frames, d_model]
  cachedK: TensorHandle,         // [B, heads, cache_len, d_head]
  cachedV: TensorHandle,
): { output: TensorHandle; newCachedK: TensorHandle; newCachedV: TensorHandle } {
  // Compute Q from chunk only
  // Compute K, V from chunk, concatenate with cached K, V
  // Run attention over [cached + chunk] keys but only for chunk queries
  // Update cache: keep last `att_context_size[0]` frames of K, V
}
```

### 7.3 — Endpointer (VAD Integration)

```typescript
// src/asr/streaming/Endpointer.ts
import { VAD } from '../../applications/speech/VAD';

export class Endpointer {
  private vad: VAD;

  processFrame(pcm: Float32Array): 'speech' | 'silence' | 'speech-end' {
    const result = this.vad.processFrame(pcm);
    // Map VAD events to endpoint decisions
  }
}
```

### 7.4 — Streaming SpeechRecognizer

Extends the recognizer with `processFrame()` streaming:

```typescript
// Streaming usage
recognizer.on('partial', ({ text }) => updateUI(text));
recognizer.on('final', ({ text, confidence, latencyMs }) => commitTranscript(text));

micStream.on('data', (frame: Float32Array) => {
  recognizer.processFrame(frame);
});
```

Internally:
1. Buffer incoming PCM frames
2. Every ~160ms (configurable chunk size), extract mel features for the new chunk
3. Run cache-aware encoder on the chunk
4. Run RNNT decoder on the new encoder output
5. Emit `partial` with current hypothesis
6. On VAD `speech-end`, emit `final` and reset decoder state

**Milestone**: Real-time streaming transcription from microphone in Node.js.

---

## Phase 8: Browser Demo (Week 12)

### 8.1 — Model Caching

```typescript
async function loadModelCached(url: string): Promise<ArrayBuffer> {
  const cache = await caches.open('asr-models');
  const cached = await cache.match(url);
  if (cached) return cached.arrayBuffer();

  const response = await fetch(url);
  cache.put(url, response.clone());
  return response.arrayBuffer();
}
```

### 8.2 — Demo Page

Add `SpeechRecognizerDemo.ts` to `demo/pages/`, using the existing `AudioInput` component and router pattern. Shows:
- Model loading progress bar
- Real-time waveform (reuse existing visualizations)
- Live transcription text with partial results
- Latency indicator
- Backend selector (wasm / webgpu / webgl)

### 8.3 — WebGPU Backend Selection

```typescript
const backend = navigator.gpu ? 'webgpu' : 'wasm';
const recognizer = new SpeechRecognizer({ backend, ... });
```

**Milestone**: Working browser demo at the Vercel deployment. Click mic, speak, see text.

---

## Phase 9: Optimization & Polish (Weeks 13–14)

- **Profile and optimize**: Use tfjs profiling (`tf.profile()`) to find which ops are slow. Consider fusing common sequences (e.g., Linear + LayerNorm + Add as one tidy block to reduce GPU dispatch overhead).
- **Decoder on CPU**: Profile whether the decode loop (both RNNT and TDT) is faster pulling data to CPU vs. running tiny GPU ops. Implement the faster path. TDT benefits more from GPU since it has fewer iterations.
- **INT8 quantization**: Quantize weights to INT8 post-training for smaller model files (120MB → 60MB for Parakeet). Dequantize to float32 at load time.
- **Error handling**: Graceful failures for missing WebGPU, model load errors, audio format mismatches.
- **Test on Nemotron 0.6B (RNNT)**: Ensure the same code works with the larger model (24 layers). Should require zero code changes — just different weights and config.
- **Test on a TDT model**: Validate TDT decode loop produces correct transcriptions and actually achieves faster inference than RNNT on the same audio.
- **Benchmark RNNT vs TDT**: Measure decoder iterations, latency, and real-time factor for both. TDT should show 2-5x fewer decoder steps.
- **Test multilingual with [Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)**: Load the multilingual weights, test with French audio. Validate that auto language detection works (the model outputs language tokens), that the larger SentencePiece vocabulary (multilingual BPE) decodes correctly for French characters (accents: é, è, ê, ë, à, ç, ù, ô, etc.), and that punctuation/capitalization output is correct.
- **French test set**: Use [Mozilla Common Voice French](https://commonvoice.mozilla.org/fr/datasets) or [FLEURS French](https://huggingface.co/datasets/google/fleurs) test split for WER measurement.
- **README, API docs, examples** — document both decoder types, multilingual usage, when to use which model
- **Benchmarks**: WER on LibriSpeech test-clean (English), Common Voice test (French), real-time factor, model load time, memory usage — for RNNT, TDT, and multilingual

---

## Summary Timeline

| Phase | What | Weeks | Effort |
|---|---|---|---|
| 0 | Groundwork: Python exports, architecture study | 1 | Light |
| 1 | Compute backend abstraction + TfjsBackend | 2 | Medium |
| 2 | Weight loading (SafeTensors + config + mapper) | 2 | Medium |
| 3 | Feature pipeline (80-band log-mel) | 3 | Medium |
| **4** | **FastConformer encoder (17 Conformer blocks)** | **4–6** | **Heavy** |
| 5 | Transducer decoders: RNNT (5a) then TDT (5b) | 7–8 | Medium |
| 6 | Integration as SpeechRecognizer application | 9 | Light |
| 7 | Cache-aware streaming | 10–11 | Heavy |
| 8 | Browser demo | 12 | Medium |
| 9 | Optimization & polish | 13–14 | Medium |

**Total: ~14 weeks.** Adding TDT does not extend the timeline because:
- The encoder (Phase 4) is identical for both — zero extra work there
- The prediction network (Phase 5.1) is shared — zero extra work
- TDT adds only a second joint network variant (~50 lines) and a second decode loop (~80 lines)
- The decode loops are structurally similar; TDT is a generalization of RNNT with a duration head

**Recommended build order within Phase 5**: Implement RNNT first (simpler, more models
available to test with), then add TDT as a second variant. The shared PredictionNetwork
and JointNetwork base class mean TDT is largely incremental.

Phases 1 and 2 can run in parallel. Phase 3 can start as soon as Phase 1 is done.
The critical path is still **Phase 4** (the encoder) — plan to spend the most time there.