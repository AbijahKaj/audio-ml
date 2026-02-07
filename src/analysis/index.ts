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
export { FFTAnalyzer, type FFTConfig } from './FFTAnalyzer';
export { MFCCAnalyzer, type MFCCConfig } from './MFCCAnalyzer';
export { PLPAnalyzer, type PLPConfig } from './PLPAnalyzer';
export { MelSpectrogramAnalyzer, type MelSpectrogramConfig } from './MelSpectrogramAnalyzer';
export { ConstantQTransformAnalyzer, type ConstantQTransformConfig } from './ConstantQTransformAnalyzer';
export { ChromaFeaturesAnalyzer, type ChromaFeaturesConfig } from './ChromaFeaturesAnalyzer';

// Spectral Feature Analyzers
export { SpectralCentroidAnalyzer, type SpectralCentroidConfig } from './SpectralCentroidAnalyzer';
export { SpectralRolloffAnalyzer, type SpectralRolloffConfig } from './SpectralRolloffAnalyzer';
export { SpectralBandwidthAnalyzer, type SpectralBandwidthConfig } from './SpectralBandwidthAnalyzer';
export { SpectralFlatnessAnalyzer, type SpectralFlatnessConfig } from './SpectralFlatnessAnalyzer';

// Time Domain Analyzers
export { ZeroCrossingRateAnalyzer, type ZeroCrossingRateConfig } from './ZeroCrossingRateAnalyzer';
export { RMSEAnalyzer, type RMSEConfig } from './RMSEAnalyzer';
export { WaveformEnvelopeAnalyzer, type WaveformEnvelopeConfig } from './WaveformEnvelopeAnalyzer';
export { AutocorrelationAnalyzer, type AutocorrelationConfig } from './AutocorrelationAnalyzer';

// Advanced Analyzers
export { LPCAnalyzer, type LPCConfig } from './LPCAnalyzer';
export { WaveletTransformAnalyzer, type WaveletTransformConfig } from './WaveletTransformAnalyzer';
