/**
 * Audio Analysis Library
 * 
 * A comprehensive collection of audio feature extractors for machine learning applications.
 * All analyzers work in both browser and Node.js environments.
 * 
 * @example
 * ```typescript
 * import { MFCCAnalyzer, FFTAnalyzer } from '@your-package/audio-analyzers';
 * 
 * const mfcc = new MFCCAnalyzer({ sampleRate: 44100 });
 * const features = mfcc.analyzeFrame(pcmData);
 * ```
 */

// Frequency Domain Analyzers
export { FFTAnalyzer, type FFTConfig } from './FFTAnalyzer.js';
export { MFCCAnalyzer, type MFCCConfig } from './MFCCAnalyzer.js';
export { PLPAnalyzer, type PLPConfig } from './PLPAnalyzer.js';
export { MelSpectrogramAnalyzer, type MelSpectrogramConfig } from './MelSpectrogramAnalyzer.js';
export { ConstantQTransformAnalyzer, type ConstantQTransformConfig } from './ConstantQTransformAnalyzer.js';
export { ChromaFeaturesAnalyzer, type ChromaFeaturesConfig } from './ChromaFeaturesAnalyzer.js';

// Spectral Feature Analyzers
export { SpectralCentroidAnalyzer, type SpectralCentroidConfig } from './SpectralCentroidAnalyzer.js';
export { SpectralRolloffAnalyzer, type SpectralRolloffConfig } from './SpectralRolloffAnalyzer.js';
export { SpectralBandwidthAnalyzer, type SpectralBandwidthConfig } from './SpectralBandwidthAnalyzer.js';
export { SpectralFlatnessAnalyzer, type SpectralFlatnessConfig } from './SpectralFlatnessAnalyzer.js';

// Time Domain Analyzers
export { ZeroCrossingRateAnalyzer, type ZeroCrossingRateConfig } from './ZeroCrossingRateAnalyzer.js';
export { RMSEAnalyzer, type RMSEConfig } from './RMSEAnalyzer.js';
export { WaveformEnvelopeAnalyzer, type WaveformEnvelopeConfig } from './WaveformEnvelopeAnalyzer.js';
export { AutocorrelationAnalyzer, type AutocorrelationConfig } from './AutocorrelationAnalyzer.js';

// Advanced Analyzers
export { LPCAnalyzer, type LPCConfig } from './LPCAnalyzer.js';
export { WaveletTransformAnalyzer, type WaveletTransformConfig } from './WaveletTransformAnalyzer.js';
