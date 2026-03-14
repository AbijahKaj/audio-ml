/**
 * Analyzers Demo Page
 * Shows all audio analyzers with visualizations
 */

import { VisualizationManager } from '../visualizations';
import { AudioInput } from '../components/AudioInput';
import { AudioInputUI } from '../components/AudioInputUI';
import {
  MFCCAnalyzer,
  PLPAnalyzer,
  ChromaFeaturesAnalyzer,
  LPCAnalyzer,
  ConstantQTransformAnalyzer,
  WaveletTransformAnalyzer,
  WaveformEnvelopeAnalyzer,
  AutocorrelationAnalyzer,
  SpectralCentroidAnalyzer,
  SpectralRolloffAnalyzer,
  SpectralBandwidthAnalyzer,
  SpectralFlatnessAnalyzer,
  ZeroCrossingRateAnalyzer,
  RMSEAnalyzer,
  MelSpectrogramAnalyzer,
  FFTAnalyzer,
} from 'audio-ml';

const fftSize = 1024;
const sampleRate = 44100;

export function createAnalyzersDemo(container: HTMLElement): () => void {
  let gridContainer: HTMLDivElement | null = null;
  let visualizationManager: VisualizationManager | null = null;
  let audioInput: AudioInput | null = null;
  let analyzerFrameSizes: Map<any, number> = new Map();

  // Create audio input component
  audioInput = new AudioInput(sampleRate);
  
  // Create audio input UI (creates UI elements)
  new AudioInputUI(container, audioInput);

  // Create grid container
  gridContainer = document.createElement("div");
  gridContainer.id = "visualization-grid";
  container.appendChild(gridContainer);

  // Setup visualizations
  setupVisualizations();

  // Connect audio input to visualizations
  const pcmHandler = (pcm: Float32Array) => {
    if (!visualizationManager) return;
    
    try {
      visualizationManager.visualizations.forEach(viz => {
        try {
          const requiredSize = analyzerFrameSizes.get(viz.analyzer);
          let frame = pcm;
          
          if (requiredSize && requiredSize !== pcm.length) {
            frame = new Float32Array(requiredSize);
            if (requiredSize > pcm.length) {
              frame.set(pcm, 0);
            } else {
              frame.set(pcm.subarray(0, requiredSize), 0);
            }
          }
          
          const result = viz.analyzer.analyzeFrame(frame);
          if (result && (Array.isArray(result) ? result.length > 0 : true)) {
            viz.visualize(result);
          }
        } catch (e) {
          console.warn(`Error visualizing ${viz.label}:`, e);
        }
      });
    } catch (e) {
      console.warn('Visualization update error:', e);
    }
  };

  audioInput.on('pcm-data', pcmHandler);

  // Cleanup function
  return () => {
    audioInput?.off('pcm-data', pcmHandler);
    audioInput?.stop();
    if (gridContainer) {
      gridContainer.remove();
    }
  };

  function setupVisualizations() {
    if (!gridContainer) return;
    
    if (visualizationManager) return;
    
    try {
      visualizationManager = new VisualizationManager(gridContainer!);
    } catch (error) {
      console.error('Error creating VisualizationManager:', error);
      return;
    }

    const mfccAnalyzer = new MFCCAnalyzer({ sampleRate, fftSize, numCoeffs: 13 });
    const plpAnalyzer = new PLPAnalyzer({ sampleRate, fftSize: 512, order: 12 });
    const chromaAnalyzer = new ChromaFeaturesAnalyzer({ sampleRate, fftSize });
    const lpcAnalyzer = new LPCAnalyzer({ sampleRate, order: 12 });
    const cqtAnalyzer = new ConstantQTransformAnalyzer({ sampleRate, fftSize: 2048 });
    const waveletAnalyzer = new WaveletTransformAnalyzer({ sampleRate, levels: 3 });
    const envelopeAnalyzer = new WaveformEnvelopeAnalyzer({ sampleRate });
    const autocorrAnalyzer = new AutocorrelationAnalyzer({ sampleRate, maxLag: 200 });
    const centroidAnalyzer = new SpectralCentroidAnalyzer({ sampleRate, fftSize });
    const rolloffAnalyzer = new SpectralRolloffAnalyzer({ sampleRate, fftSize });
    const bandwidthAnalyzer = new SpectralBandwidthAnalyzer({ sampleRate, fftSize });
    const flatnessAnalyzer = new SpectralFlatnessAnalyzer({ sampleRate, fftSize });
    const zcrAnalyzer = new ZeroCrossingRateAnalyzer({ sampleRate });
    const rmseAnalyzer = new RMSEAnalyzer({ sampleRate });
    const melSpectrogramAnalyzer = new MelSpectrogramAnalyzer({ sampleRate, fftSize, melBands: 40 });
    const fftAnalyzer = new FFTAnalyzer({ sampleRate, fftSize });

    analyzerFrameSizes.clear();
    analyzerFrameSizes.set(plpAnalyzer, 512);
    analyzerFrameSizes.set(cqtAnalyzer, 2048);
    analyzerFrameSizes.set(melSpectrogramAnalyzer, fftSize);
    analyzerFrameSizes.set(mfccAnalyzer, fftSize);
    analyzerFrameSizes.set(fftAnalyzer, fftSize);

    const colors = [
      '#00ff88', '#ff8800', '#88ff00', '#ff0088', '#0088ff',
      '#00ffff', '#ffff00', '#ff00ff', '#00ffff', '#ff8800',
      '#8800ff', '#00ff88', '#ff0088', '#88ff00', '#ffaa00'
    ];

    const analyzers = [
      { analyzer: mfccAnalyzer, label: 'MFCC', width: 400, height: 200 },
      { analyzer: plpAnalyzer, label: 'PLP', width: 400, height: 200 },
      { analyzer: chromaAnalyzer, label: 'Chroma Features', width: 400, height: 150 },
      { analyzer: lpcAnalyzer, label: 'LPC', width: 400, height: 200 },
      { analyzer: cqtAnalyzer, label: 'Constant-Q Transform', width: 400, height: 250 },
      { analyzer: melSpectrogramAnalyzer, label: 'Mel Spectrogram', width: 400, height: 250 },
      { analyzer: fftAnalyzer, label: 'FFT', width: 400, height: 200 },
      { analyzer: waveletAnalyzer, label: 'Wavelet Transform', width: 400, height: 200 },
      { analyzer: envelopeAnalyzer, label: 'Waveform Envelope', width: 400, height: 150 },
      { analyzer: autocorrAnalyzer, label: 'Autocorrelation', width: 400, height: 200 },
      { analyzer: centroidAnalyzer, label: 'Spectral Centroid', width: 400, height: 150 },
      { analyzer: rolloffAnalyzer, label: 'Spectral Rolloff', width: 400, height: 150 },
      { analyzer: bandwidthAnalyzer, label: 'Spectral Bandwidth', width: 400, height: 150 },
      { analyzer: flatnessAnalyzer, label: 'Spectral Flatness', width: 400, height: 150 },
      { analyzer: zcrAnalyzer, label: 'Zero Crossing Rate', width: 400, height: 150 },
      { analyzer: rmseAnalyzer, label: 'RMSE', width: 400, height: 150 },
    ];

    analyzers.forEach((item, index) => {
      try {
        const canvas = document.createElement('canvas');
        canvas.width = item.width;
        canvas.height = item.height;
        visualizationManager!.addVisualization(
          item.analyzer,
          canvas,
          item.label,
          {
            color: colors[index % colors.length],
            backgroundColor: '#000000',
            width: item.width,
            height: item.height
          }
        );
      } catch (error) {
        console.error(`Error adding visualization for ${item.label}:`, error);
      }
    });
  }
}
