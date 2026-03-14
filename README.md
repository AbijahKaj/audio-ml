# Audio ML - Audio Analysis for Machine Learning

A JavaScript/TypeScript library for real-time audio feature extraction and processing. Works in both browsers and Node.js.

## Installation

```bash
npm install audio-ml
# or
yarn add audio-ml
# or
pnpm add audio-ml
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

## Use Cases

- **Speech recognition**: MFCC and PLP for acoustic modeling, Spectral features for phone classification
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
# Run the interactive demo
cd demo && yarn install && yarn dev
```

The demo includes live visualizations of all 16 analyzers plus interactive pages for each application.

## Contributing

To add a new analyzer, create a class in `src/analysis/` implementing `analyzeFrame(pcm: Float32Array)` and export it from `src/analysis/index.ts`.

To add a new application, extend `BaseApplication` in `src/applications/`, implement `processFrame()`, and export from `src/applications/index.ts`.

## License

MIT - See [LICENSE](LICENSE) file for details
