import { VisualizationManager } from './visualizations';
import { MFCCAnalyzer } from './analysis/MFCCAnalyzer';
import { PLPAnalyzer } from './analysis/PLPAnalyzer';
import { ChromaFeaturesAnalyzer } from './analysis/ChromaFeaturesAnalyzer';
import { LPCAnalyzer } from './analysis/LPCAnalyzer';
import { ConstantQTransformAnalyzer } from './analysis/ConstantQTransformAnalyzer';
import { WaveletTransformAnalyzer } from './analysis/WaveletTransformAnalyzer';
import { WaveformEnvelopeAnalyzer } from './analysis/WaveformEnvelopeAnalyzer';
import { AutocorrelationAnalyzer } from './analysis/AutocorrelationAnalyzer';
import { SpectralCentroidAnalyzer } from './analysis/SpectralCentroidAnalyzer';
import { SpectralRolloffAnalyzer } from './analysis/SpectralRolloffAnalyzer';
import { SpectralBandwidthAnalyzer } from './analysis/SpectralBandwidthAnalyzer';
import { SpectralFlatnessAnalyzer } from './analysis/SpectralFlatnessAnalyzer';
import { ZeroCrossingRateAnalyzer } from './analysis/ZeroCrossingRateAnalyzer';
import { RMSEAnalyzer } from './analysis/RMSEAnalyzer';
import { MelSpectrogramAnalyzer } from './analysis/MelSpectrogramAnalyzer';
import { FFTAnalyzer } from './analysis/FFTAnalyzer';

// Module-level variables
let gridContainer: HTMLDivElement | null = null;
let buttonContainer: HTMLDivElement | null = null;
let recordButton: HTMLButtonElement | null = null;
let mediaRecorder: MediaRecorder | null = null;
let audioCtx: AudioContext | null = null;
let processor: ScriptProcessorNode | null = null;
let visualizationManager: VisualizationManager | null = null;
let stream: MediaStream | null = null;
let analyzerFrameSizes: Map<any, number> = new Map();

const fftSize = 1024;
const sampleRate = 44100;

// Wait for DOM to be ready
document.addEventListener('DOMContentLoaded', () => {
  const appDiv = document.getElementById("app");
  if (!appDiv) {
    console.error("App div not found");
    return;
  }

  // Create button container
  buttonContainer = document.createElement("div");
  buttonContainer.id = "button-container";
  appDiv.appendChild(buttonContainer);

  // Create button
  recordButton = document.createElement("button");
  recordButton.textContent = "Start Recording";
  recordButton.id = "recordButton";
  buttonContainer.appendChild(recordButton);

  // Create grid container
  gridContainer = document.createElement("div");
  gridContainer.id = "visualization-grid";
  appDiv.appendChild(gridContainer);

  // Setup visualizations immediately so grid is visible
  setupVisualizations();
  
  // Set up button event listener
  if (recordButton) {
    recordButton.addEventListener("click", handleRecordButtonClick);
  }
});

function setupVisualizations() {
  if (!gridContainer) {
    console.error('Grid container not found');
    return;
  }
  
  // Only setup if not already done
  if (visualizationManager) {
    console.log('Visualizations already set up');
    return;
  }
  
  try {
    visualizationManager = new VisualizationManager(gridContainer);
    console.log('VisualizationManager created');
  } catch (error) {
    console.error('Error creating VisualizationManager:', error);
    return;
  }

  // Create all analyzers
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

  // Store frame sizes for analyzers that need specific sizes
  analyzerFrameSizes.clear();
  analyzerFrameSizes.set(plpAnalyzer, 512);
  analyzerFrameSizes.set(cqtAnalyzer, 2048);
  analyzerFrameSizes.set(melSpectrogramAnalyzer, fftSize); // MFCC and Mel need fftSize
  analyzerFrameSizes.set(mfccAnalyzer, fftSize);
  analyzerFrameSizes.set(fftAnalyzer, fftSize);

  // Add visualizations with different colors
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
  
  console.log(`Set up ${analyzers.length} visualizations`);
}

async function handleRecordButtonClick() {
  if (!recordButton || !buttonContainer) return;
  
  if (!mediaRecorder || mediaRecorder.state === "inactive") {
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          sampleRate: sampleRate,
          echoCancellation: false,
          noiseSuppression: false,
        },
      });

      audioCtx = new AudioContext({ sampleRate });
      const source = audioCtx.createMediaStreamSource(stream);

      // Use ScriptProcessorNode to get PCM samples
      processor = audioCtx.createScriptProcessor(fftSize, 1, 1);
      source.connect(processor);
      processor.connect(audioCtx.destination);

      processor.onaudioprocess = (event) => {
        const input = event.inputBuffer.getChannelData(0);
        if (!visualizationManager) return;
        
        try {
          // Update each visualization with appropriately sized frame
          visualizationManager.visualizations.forEach(viz => {
            try {
              const requiredSize = analyzerFrameSizes.get(viz.analyzer);
              let frame = input;
              
              if (requiredSize && requiredSize !== input.length) {
                // Pad or truncate to required size
                frame = new Float32Array(requiredSize);
                if (requiredSize > input.length) {
                  frame.set(input, 0);
                  // Pad with zeros (already zero-filled)
                } else {
                  frame.set(input.subarray(0, requiredSize), 0);
                }
              }
              
              const result = viz.analyzer.analyzeFrame(frame);
              if (result && (Array.isArray(result) ? result.length > 0 : true)) {
                viz.visualize(result);
              }
            } catch (e) {
              // Log errors for debugging
              console.warn(`Error visualizing ${viz.label}:`, e);
            }
          });
        } catch (e) {
          console.warn('Visualization update error:', e);
        }
      };

      mediaRecorder = new MediaRecorder(stream);
      const audioChunks: Blob[] = [];
      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunks.push(event.data);
        }
      };
      mediaRecorder.onstop = () => {
        const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
        const audioUrl = URL.createObjectURL(audioBlob);
        const audio = document.createElement('audio');
        audio.controls = true;
        audio.src = audioUrl;
        audio.style.marginLeft = '1rem';
        buttonContainer!.appendChild(audio);
      };
      mediaRecorder.start();
      recordButton.textContent = "Stop Recording";
    } catch (err) {
      alert("Microphone access denied or not available: " + err);
      console.error(err);
    }
  } else if (mediaRecorder.state === "recording") {
    mediaRecorder.stop();
    recordButton.textContent = "Start Recording";
    
    // Cleanup
    if (processor) {
      processor.disconnect();
      processor = null;
    }
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
      stream = null;
    }
    if (audioCtx) {
      audioCtx.close();
      audioCtx = null;
    }
  }
}
