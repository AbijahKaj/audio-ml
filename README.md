# Audio ML - Audio Analysis for Machine Learning

A JavaScript/TypeScript library for real-time audio feature extraction and processing. Works in both browsers and Node.js.

## Installation

Core library (analyzers and applications such as VAD):

```bash
npm install audio-ml
# or
yarn add audio-ml
# or
pnpm add audio-ml
```

**FastConformer ASR** (TensorFlow.js inference) lives in a separate package and lists `audio-ml` as a peer dependency:

```bash
npm install audio-ml @audio-ml/asr
# or
yarn add audio-ml @audio-ml/asr
```

## Demo

https://github.com/user-attachments/assets/aae5ff8c-120b-4c6c-a4d4-7348dacc3ca0

## Analyzers

16 low-level audio analyzers, all sharing the same interface: `analyzer.analyzeFrame(pcm: Float32Array)`.

```typescript
import { FFTAnalyzer, MFCCAnalyzer } from 'audio-ml';

const fft = new FFTAnalyzer({ sampleRate: 44100, fftSize: 1024 });
const mfcc = new MFCCAnalyzer({ sampleRate: 44100 });

const spectrum = fft.analyzeFrame(pcmFrame);    // Float32Array
const features = mfcc.analyzeFrame(pcmFrame);   // number[]
```

| Analyzer | Output | Description |
|----------|--------|-------------|
| FFT | `Float32Array` | Magnitude spectrum |
| MFCC | `number[]` | 13 mel-frequency cepstral coefficients |
| PLP | `number[]` | Perceptual linear prediction |
| Mel Spectrogram | `number[]` | Mel-scaled power spectrum |
| Constant-Q Transform | `Float32Array` | Log-spaced frequency analysis |
| Chroma Features | `number[]` | 12-tone pitch class distribution |
| Spectral Centroid | `number` | Frequency center of mass (Hz) |
| Spectral Rolloff | `number` | 85th-percentile frequency (Hz) |
| Spectral Bandwidth | `number` | Spectral spread around centroid (Hz) |
| Spectral Flatness | `number` | Noise-like vs tonal content (0–1) |
| Zero Crossing Rate | `number` | Rate of sign changes |
| RMSE | `number` | Root mean square energy |
| Waveform Envelope | `Float32Array` | Amplitude envelope |
| Autocorrelation | `Float32Array` | Periodicity / pitch detection |
| LPC | `number[]` | Linear predictive coding coefficients |
| Wavelet Transform | `Float32Array[]` | Multi-level time-frequency decomposition |

## Applications

Higher-level tools built on top of the analyzers. Import from `audio-ml/applications`. All applications extend `BaseApplication` with an event-driven API: call `processFrame()` per audio frame, listen for events.

```typescript
import { VAD, AudioDenoiser, VoicemailBeepDetector } from 'audio-ml/applications';
```

### Voice Activity Detection (VAD)

Detects speech vs silence by combining RMSE, Zero Crossing Rate, Spectral Flatness, and Spectral Centroid with weighted scoring and temporal smoothing.

```typescript
const vad = new VAD({ sampleRate: 44100 });

vad.on('speech-start', ({ confidence }) => console.log('Speaking', confidence));
vad.on('speech-end', ({ confidence }) => console.log('Silent', confidence));

// Per frame
const result = vad.processFrame(pcm); // { isSpeech, confidence, features }
```

### Audio Denoiser

Removes background noise via spectral subtraction. Automatically estimates the noise profile from initial silence using RMSE and Spectral Flatness, then subtracts it in the frequency domain.

```typescript
const denoiser = new AudioDenoiser({ sampleRate: 44100, fftSize: 2048 });

denoiser.on('noise-estimated', () => console.log('Noise profile ready'));
denoiser.on('denoised-frame', ({ audio, snr }) => { /* clean audio */ });

const { audio, snr, noiseReduction } = denoiser.processFrame(pcm);
```

### Voicemail Beep Detector

Detects tonal beeps using FFT peak detection across configurable frequency ranges, with sustained-tone tracking and duration filtering.

```typescript
const detector = new VoicemailBeepDetector({
  sampleRate: 44100,
  fftSize: 2048,
  frequencyRanges: [
    { min: 400, max: 500, name: 'Low beep' },
    { min: 900, max: 1100, name: 'Mid beep' },
  ]
});

detector.on('beep-detected', ({ frequency, duration, confidence }) => {
  console.log(`Beep at ${frequency} Hz`);
});

detector.processFrame(pcm);
```

## Automatic speech recognition (`@audio-ml/asr`)

End-to-end **FastConformer** ASR (feature extraction, encoder, RNNT/TDT greedy decode, optional streaming) runs in the browser or Node via TensorFlow.js. Import **`FastConformerASR`** from **`@audio-ml/asr`**. It extends the same **`BaseApplication`** / event model as the apps above, but depends on **`audio-ml/applications`** internally (for example VAD-backed endpointing).

```typescript
import { FastConformerASR, type ASRResult } from '@audio-ml/asr';

const asr = new FastConformerASR({
  sampleRate: 16_000,
  modelPath: '/models/weights.safetensors',
  configPath: '/models/model_config.json',
  vocabPath: '/models/vocab.json',
  backend: 'webgpu', // browser: prefer WebGPU; use 'webgl' if unavailable (avoid 'cpu' / 'wasm' for large models)
  streaming: true,
});

await asr.load();

asr.on('partial', (p) => console.log('partial', p.text));
asr.on('final', (r: ASRResult) => console.log('final', r.text));

// Per frame (see package exports for batch helpers such as transcribe())
asr.processFrame(pcmFrame);
```

In **Node.js**, install `@tensorflow/tfjs-node` (optional peer) and use `backend: 'tensorflow'` for native acceleration; otherwise use `'cpu'` (pure JS).

The package also exports lower-level pieces (`FastConformerEncoder`, decoders, `FeaturePipeline`, `TfjsBackend`, etc.) if you want to compose your own pipeline.

## Use Cases

- **Speech recognition (end-to-end)**: `@audio-ml/asr` with **`FastConformerASR`** for in-browser or Node ASR with exported NeMo-compatible weights
- **Speech recognition (features only)**: MFCC and PLP for acoustic modeling, spectral features for phone classification
- **Speaker identification**: Voiceprint extraction via MFCC, LPC, and Spectral Centroid/Bandwidth
- **Voice activity detection**: VAD application, or build your own with RMSE, ZCR, and Spectral Flatness
- **Noise reduction**: AudioDenoiser application for real-time spectral subtraction
- **Telephony**: VoicemailBeepDetector for detecting end-of-greeting tones in voicemail systems
- **Music analysis**: Chroma Features for chord/key detection, Autocorrelation for tempo, CQT for pitch tracking
- **Genre / mood classification**: Combine MFCC, Spectral Rolloff, Bandwidth, and Flatness as ML feature vectors
- **Onset detection**: Waveform Envelope and Spectral Flatness for detecting note/event boundaries

## Platform Support

- **Browser**: Modern browsers with Web Audio API. Works with Vite, Webpack, Rollup, etc.
- **Node.js**: 18.0.0+. Pair with audio decoding libraries (node-wav, audio-decode) for file processing.

## Development

```bash
# Build the main library and @audio-ml/asr (required for the ASR demo page)
yarn build:all

# Run the interactive demo
cd demo && yarn install && yarn dev
```

The demo includes live visualizations of all 16 analyzers, interactive pages for each application, and a **Speech recognizer** page powered by **`@audio-ml/asr`**.

Repository layout:

- **`src/`** — `audio-ml` package (analyzers under `src/analysis/`, applications under `src/applications/`)
- **`packages/asr/`** — `@audio-ml/asr` (FastConformer stack and **`FastConformerASR`**)

## Contributing

To add a new analyzer, create a class in `src/analysis/` implementing `analyzeFrame(pcm: Float32Array)` and export it from `src/analysis/index.ts`.

To add a new application (other than ASR), extend `BaseApplication` in `src/applications/`, implement `processFrame()`, and export from `src/applications/index.ts`.

ASR and related inference code belong in **`packages/asr/`**; export new public APIs from **`packages/asr/src/index.ts`**.

## License

MIT - See [LICENSE](LICENSE) file for details
