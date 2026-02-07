# Audio ML - Audio Analysis for Machine Learning

A comprehensive JavaScript/TypeScript library for real-time audio feature extraction, designed for machine learning applications, particularly voice AI systems.

## 📦 Installation

```bash
npm install audio-ml
# or
yarn add audio-ml
# or
pnpm add audio-ml
```

**Works in both Web and Node.js environments!** This package is designed to be universal - use it in your browser-based applications or in Node.js server-side applications.

## 🎬 Demo

https://github.com/user-attachments/assets/aae5ff8c-120b-4c6c-a4d4-7348dacc3ca0


## Overview

This project provides a complete toolkit for analyzing audio signals in real-time, extracting various features that are essential for machine learning models in speech recognition, speaker identification, music information retrieval, and voice AI applications.

## Features

### 🎯 Real-Time Audio Analysis
- Process audio from microphone or audio playback in real-time
- Extract multiple audio features simultaneously
- Visualize all features in a responsive grid layout

### 📊 16 Audio Analyzers

#### Frequency Domain Features
- **FFT** - Fast Fourier Transform magnitude spectrum
- **MFCC** - Mel-Frequency Cepstral Coefficients (13 coefficients)
- **PLP** - Perceptual Linear Prediction
- **Mel Spectrogram** - Mel-scaled power spectrum
- **Constant-Q Transform** - Logarithmically spaced frequency analysis
- **Chroma Features** - 12-tone pitch class representation

#### Spectral Features
- **Spectral Centroid** - Frequency "center of mass"
- **Spectral Rolloff** - Frequency below which 85% of energy lies
- **Spectral Bandwidth** - Spread of spectrum around centroid
- **Spectral Flatness** - Measure of noise-like vs tone-like content

#### Time Domain Features
- **Zero Crossing Rate** - Rate of sign changes
- **RMSE** - Root Mean Square Energy
- **Waveform Envelope** - Amplitude envelope tracking
- **Autocorrelation** - Periodicity and pitch detection

#### Advanced Features
- **LPC** - Linear Predictive Coding coefficients
- **Wavelet Transform** - Multi-level time-frequency decomposition

### 🎨 Interactive Visualizations
- Real-time canvas-based visualizations for each analyzer
- Responsive grid layout (up to 4 columns)
- Info tooltips with detailed explanations and resources
- Scrolling spectrograms for time-frequency analysis

## Getting Started

### Using the Package

The `audio-ml` package can be used in both **web browsers** and **Node.js** environments:

#### Web Browser Usage

```typescript
import { FFTAnalyzer, MFCCAnalyzer } from 'audio-ml';

// Create analyzers
const fftAnalyzer = new FFTAnalyzer({ 
  sampleRate: 44100, 
  fftSize: 1024 
});

// Use with Web Audio API
const audioContext = new AudioContext();
const processor = audioContext.createScriptProcessor(1024, 1, 1);

processor.onaudioprocess = (event) => {
  const pcm = event.inputBuffer.getChannelData(0);
  const spectrum = fftAnalyzer.analyzeFrame(pcm);
  // Process your features...
};
```

#### Node.js Usage

```typescript
import { FFTAnalyzer, MFCCAnalyzer } from 'audio-ml';
import { readFileSync } from 'fs';
import { decode } from 'audio-decode'; // or similar audio decoder

// Load and decode audio file
const audioBuffer = await decode(readFileSync('audio.wav'));

// Create analyzer
const mfccAnalyzer = new MFCCAnalyzer({ 
  sampleRate: audioBuffer.sampleRate 
});

// Process audio frames
const frameSize = 1024;
for (let i = 0; i < audioBuffer.length; i += frameSize) {
  const frame = audioBuffer.getChannelData(0).subarray(i, i + frameSize);
  const features = mfccAnalyzer.analyzeFrame(frame);
  // Use features for ML models...
}
```

### Development

```bash
npm run dev
# or
yarn dev
```

Open your browser and navigate to the local development server (typically `http://localhost:5173`).

### Usage

1. Click "Start Recording" to begin capturing audio from your microphone
2. All 16 analyzers will update in real-time as you speak
3. Click the ⓘ icon next to any analyzer name to learn more about it
4. Click "Stop Recording" to end the session

## Architecture

### Analyzers (`src/analysis/`)
Each analyzer is a self-contained class that:
- Takes PCM audio frames as input
- Returns feature vectors or scalar values
- Handles its own FFT and signal processing

### Visualizations (`src/visualizations/`)
- **Base classes**: `BaseVisualizer`, `ArrayVisualizer`, `ScalarVisualizer`
- **Visualizer functions**: Specific drawing functions for each analyzer type
- **VisualizationManager**: Manages multiple visualizations and updates them in real-time
- **Info system**: Tooltips with detailed information about each analyzer

### Main Application (`src/main.ts`)
- Sets up audio capture from microphone
- Creates and manages all analyzers
- Handles frame size differences between analyzers
- Updates visualizations in real-time

## API Reference

### Basic Usage

```typescript
import { FFTAnalyzer, MFCCAnalyzer } from 'audio-ml';

// Create analyzers
const fftAnalyzer = new FFTAnalyzer({ 
  sampleRate: 44100, 
  fftSize: 1024 
});

const mfccAnalyzer = new MFCCAnalyzer({ 
  sampleRate: 44100 
});

// Analyze a frame
const pcmFrame = new Float32Array(1024); // Your audio data
const spectrum = fftAnalyzer.analyzeFrame(pcmFrame);
const mfccFeatures = mfccAnalyzer.analyzeFrame(pcmFrame);

// Use the features for your ML model or further processing
console.log('FFT Spectrum:', spectrum);
console.log('MFCC Features:', mfccFeatures);
```

## Analyzers Reference

| Analyzer | Output Type | Frame Size | Description |
|----------|-------------|------------|-------------|
| FFT | `Float32Array` | 1024 | Raw frequency spectrum |
| MFCC | `number[]` | 1024 | 13 cepstral coefficients |
| PLP | `number[]` | 512 | Perceptual linear prediction |
| Chroma | `number[]` | 1024 | 12 pitch classes |
| LPC | `number[]` | Any | Linear prediction coefficients |
| CQT | `Float32Array` | 2048 | Constant-Q transform |
| Wavelet | `Float32Array[]` | Any | Multi-level decomposition |
| Envelope | `Float32Array` | Any | Amplitude envelope |
| Autocorr | `Float32Array` | Any | Autocorrelation function |
| Centroid | `number` | 1024 | Spectral centroid (Hz) |
| Rolloff | `number` | 1024 | Spectral rolloff (Hz) |
| Bandwidth | `number` | 1024 | Spectral bandwidth (Hz) |
| Flatness | `number` | 1024 | Spectral flatness (0-1) |
| ZCR | `number` | Any | Zero crossing rate |
| RMSE | `number` | Any | Root mean square energy |
| Mel Spectrogram | `number[]` | 1024 | Mel-scaled energies |

## Use Cases

### Voice AI Applications
- **Speech Recognition**: MFCC, PLP, and spectral features for acoustic modeling
- **Speaker Identification**: Voiceprint extraction using MFCC, LPC, and spectral features
- **Voice Activity Detection**: ZCR, RMSE, and spectral flatness for silence detection
- **Emotion Recognition**: Spectral features and prosodic features

### Music Information Retrieval
- **Chord Recognition**: Chroma features
- **Key Detection**: Chroma features and spectral analysis
- **Tempo Estimation**: Autocorrelation
- **Genre Classification**: Multiple spectral and temporal features

### Audio Processing
- **Noise Reduction**: Spectral analysis and filtering
- **Pitch Detection**: Autocorrelation and CQT
- **Onset Detection**: Envelope and spectral features

## Dependencies

- **fft.js** - Fast Fourier Transform implementation
- **TypeScript** - Type-safe JavaScript
- **Vite** - Build tool and dev server

## Platform Support

### Web Browser
- Modern browsers with Web Audio API support
- Microphone access required for real-time analysis
- Canvas API for visualizations
- Works with ES modules and bundlers (Vite, Webpack, Rollup, etc.)

### Node.js
- Node.js 18.0.0 or higher
- Works with CommonJS and ES modules
- Compatible with audio decoding libraries (node-wav, audio-decode, etc.)
- Perfect for server-side audio processing and ML pipelines

## Contributing

This project is designed to be extensible. To add a new analyzer:

1. Create a new analyzer class in `src/analysis/`
2. Implement the `analyzeFrame(pcm: Float32Array)` method
3. Add a visualization function in `src/visualizations/analyzerVisualizers.ts`
4. Register it in `VisualizationManager.getVisualizer()`
5. Add info to `src/visualizations/analyzerInfo.ts`

## License

MIT - See [LICENSE](LICENSE) file for details

## Resources

- [Web Audio API](https://developer.mozilla.org/en-US/docs/Web/API/Web_Audio_API)
- [FFT.js Documentation](https://github.com/indutny/fft.js)
