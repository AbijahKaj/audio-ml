/**
 * Specific visualizers for each analyzer type
 */

import { ArrayVisualizer, ScalarVisualizer } from './base';
import type { VisualizationOptions } from './base';

// Array-based analyzers
export function visualizeMFCC(
  canvas: HTMLCanvasElement,
  data: number[],
  options?: VisualizationOptions
) {
  const viz = new ArrayVisualizer(canvas, { ...options, color: options?.color || '#00ff88' });
  viz.drawBars(data, true);
}

export function visualizePLP(
  canvas: HTMLCanvasElement,
  data: number[],
  options?: VisualizationOptions
) {
  const viz = new ArrayVisualizer(canvas, { ...options, color: options?.color || '#ff8800' });
  viz.drawLine(data, true);
}

export function visualizeChroma(
  canvas: HTMLCanvasElement,
  data: number[],
  options?: VisualizationOptions
) {
  const viz = new ArrayVisualizer(canvas, { ...options, color: options?.color || '#88ff00' });
  // Chroma has 12 bins, draw as circular or bars
  const chromaNames = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
  viz.drawBars(data, false); // Already normalized (0-1)
  
  // Add labels
  const ctx = canvas.getContext('2d');
  if (ctx && data.length === 12) {
    ctx.fillStyle = options?.color || '#88ff00';
    ctx.font = '10px monospace';
    ctx.textAlign = 'center';
    const barWidth = canvas.width / 12;
    for (let i = 0; i < 12; i++) {
      ctx.fillText(chromaNames[i], i * barWidth + barWidth / 2, canvas.height - 5);
    }
  }
}

export function visualizeLPC(
  canvas: HTMLCanvasElement,
  data: number[],
  options?: VisualizationOptions
) {
  const viz = new ArrayVisualizer(canvas, { ...options, color: options?.color || '#ff0088' });
  viz.drawLine(data, true);
}

// Store visualizer instances for stateful visualizations
const visualizerInstances = new WeakMap<HTMLCanvasElement, ArrayVisualizer>();

function getOrCreateVisualizer(
  canvas: HTMLCanvasElement,
  options?: VisualizationOptions,
  defaultColor?: string
): ArrayVisualizer {
  let viz = visualizerInstances.get(canvas);
  if (!viz) {
    viz = new ArrayVisualizer(canvas, { ...options, color: options?.color || defaultColor || '#00ff00' });
    visualizerInstances.set(canvas, viz);
  }
  return viz;
}

export function visualizeCQT(
  canvas: HTMLCanvasElement,
  data: Float32Array,
  options?: VisualizationOptions
) {
  const viz = getOrCreateVisualizer(canvas, options, '#0088ff');
  viz.drawSpectrogram(data);
}

export function visualizeMelSpectrogram(
  canvas: HTMLCanvasElement,
  data: number[],
  options?: VisualizationOptions
) {
  const viz = getOrCreateVisualizer(canvas, options, '#ffaa00');
  // Use spectrogram for scrolling time-frequency display
  viz.drawSpectrogram(new Float32Array(data));
}

export function visualizeFFT(
  canvas: HTMLCanvasElement,
  data: Float32Array,
  options?: VisualizationOptions
) {
  const viz = getOrCreateVisualizer(canvas, options, '#00ff00');
  viz.drawBars(data, true);
}

export function visualizeWavelet(
  canvas: HTMLCanvasElement,
  data: Float32Array[],
  options?: VisualizationOptions
) {
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  
  ctx.fillStyle = options?.backgroundColor || '#000000';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  
  const levels = data.length;
  if (levels === 0) return;
  
  const levelHeight = canvas.height / levels;
  const color = options?.color || '#00ffff';
  
  for (let level = 0; level < levels; level++) {
    const coeffs = data[level];
    const len = coeffs.length;
    if (len === 0) continue;
    
    const y = level * levelHeight;
    const stepX = canvas.width / len;
    
    ctx.strokeStyle = color;
    ctx.lineWidth = 1;
    ctx.beginPath();
    
    const values = Array.from(coeffs);
    const min = Math.min(...values);
    const max = Math.max(...values);
    
    for (let i = 0; i < len; i++) {
      const normalized = max === min ? 0 : (values[i] - min) / (max - min);
      const x = i * stepX;
      const yPos = y + levelHeight / 2 - (normalized - 0.5) * levelHeight * 0.8;
      if (i === 0) {
        ctx.moveTo(x, yPos);
      } else {
        ctx.lineTo(x, yPos);
      }
    }
    ctx.stroke();
  }
}

export function visualizeEnvelope(
  canvas: HTMLCanvasElement,
  data: Float32Array,
  options?: VisualizationOptions
) {
  const viz = new ArrayVisualizer(canvas, { ...options, color: options?.color || '#ffff00' });
  viz.drawLine(data, true);
}

export function visualizeAutocorrelation(
  canvas: HTMLCanvasElement,
  data: Float32Array,
  options?: VisualizationOptions
) {
  const viz = new ArrayVisualizer(canvas, { ...options, color: options?.color || '#ff00ff' });
  viz.drawLine(data, true);
}

// Scalar analyzers
export function visualizeSpectralCentroid(
  canvas: HTMLCanvasElement,
  value: number,
  options?: VisualizationOptions & { maxHistoryLength?: number }
) {
  const viz = new ScalarVisualizer(canvas, { ...options, color: options?.color || '#00ffff' });
  // Typical range: 0 to sampleRate/2 Hz
  const maxFreq = 22050; // Default for 44.1kHz
  viz.drawTimeSeries(value, 0, maxFreq);
}

export function visualizeSpectralRolloff(
  canvas: HTMLCanvasElement,
  value: number,
  options?: VisualizationOptions & { maxHistoryLength?: number }
) {
  const viz = new ScalarVisualizer(canvas, { ...options, color: options?.color || '#ff8800' });
  const maxFreq = 22050;
  viz.drawTimeSeries(value, 0, maxFreq);
}

export function visualizeSpectralBandwidth(
  canvas: HTMLCanvasElement,
  value: number,
  options?: VisualizationOptions & { maxHistoryLength?: number }
) {
  const viz = new ScalarVisualizer(canvas, { ...options, color: options?.color || '#8800ff' });
  const maxFreq = 22050;
  viz.drawTimeSeries(value, 0, maxFreq);
}

export function visualizeSpectralFlatness(
  canvas: HTMLCanvasElement,
  value: number,
  options?: VisualizationOptions & { maxHistoryLength?: number }
) {
  const viz = new ScalarVisualizer(canvas, { ...options, color: options?.color || '#00ff88' });
  viz.drawTimeSeries(value, 0, 1); // Flatness is 0-1
}

export function visualizeZeroCrossingRate(
  canvas: HTMLCanvasElement,
  value: number,
  options?: VisualizationOptions & { maxHistoryLength?: number }
) {
  const viz = new ScalarVisualizer(canvas, { ...options, color: options?.color || '#ff0088' });
  viz.drawTimeSeries(value, 0, 0.5); // ZCR typically 0-0.5
}

export function visualizeRMSE(
  canvas: HTMLCanvasElement,
  value: number,
  options?: VisualizationOptions & { maxHistoryLength?: number }
) {
  const viz = new ScalarVisualizer(canvas, { ...options, color: options?.color || '#88ff00' });
  viz.drawTimeSeries(value); // Auto-scale
}
