/**
 * VisualizationManager - Helper class to manage multiple analyzer visualizations
 */

import {
  visualizeMFCC,
  visualizePLP,
  visualizeChroma,
  visualizeLPC,
  visualizeCQT,
  visualizeWavelet,
  visualizeEnvelope,
  visualizeAutocorrelation,
  visualizeSpectralCentroid,
  visualizeSpectralRolloff,
  visualizeSpectralBandwidth,
  visualizeSpectralFlatness,
  visualizeZeroCrossingRate,
  visualizeRMSE,
  visualizeMelSpectrogram,
  visualizeFFT,
} from './analyzerVisualizers';

import { MFCCAnalyzer } from '../../src/analysis/MFCCAnalyzer';
import { PLPAnalyzer } from '../../src/analysis/PLPAnalyzer';
import { ChromaFeaturesAnalyzer } from '../../src/analysis/ChromaFeaturesAnalyzer';
import { LPCAnalyzer } from '../../src/analysis/LPCAnalyzer';
import { ConstantQTransformAnalyzer } from '../../src/analysis/ConstantQTransformAnalyzer';
import { WaveletTransformAnalyzer } from '../../src/analysis/WaveletTransformAnalyzer';
import { WaveformEnvelopeAnalyzer } from '../../src/analysis/WaveformEnvelopeAnalyzer';
import { AutocorrelationAnalyzer } from '../../src/analysis/AutocorrelationAnalyzer';
import { SpectralCentroidAnalyzer } from '../../src/analysis/SpectralCentroidAnalyzer';
import { SpectralRolloffAnalyzer } from '../../src/analysis/SpectralRolloffAnalyzer';
import { SpectralBandwidthAnalyzer } from '../../src/analysis/SpectralBandwidthAnalyzer';
import { SpectralFlatnessAnalyzer } from '../../src/analysis/SpectralFlatnessAnalyzer';
import { ZeroCrossingRateAnalyzer } from '../../src/analysis/ZeroCrossingRateAnalyzer';
import { RMSEAnalyzer } from '../../src/analysis/RMSEAnalyzer';
import { MelSpectrogramAnalyzer } from '../../src/analysis/MelSpectrogramAnalyzer';
import { FFTAnalyzer } from '../../src/analysis/FFTAnalyzer';
import { analyzerInfoMap } from './analyzerInfo';

export interface AnalyzerVisualization {
  analyzer: any;
  canvas: HTMLCanvasElement;
  visualize: (data: any) => void;
  visualizer?: any; // Store visualizer instance for stateful visualizations
  label?: string;
}

export class VisualizationManager {
  public visualizations: AnalyzerVisualization[] = [];
  private container: HTMLElement;

  constructor(container: HTMLElement) {
    this.container = container;
  }

  /**
   * Add an analyzer visualization
   */
  addVisualization(
    analyzer: any,
    canvas: HTMLCanvasElement,
    label?: string,
    options?: { width?: number; height?: number; color?: string; backgroundColor?: string }
  ) {
    const viz: AnalyzerVisualization = {
      analyzer,
      canvas,
      visualize: this.getVisualizer(analyzer, canvas, options),
      label,
    };

    // Wrap label and canvas in a container div for grid layout
    const containerDiv = document.createElement('div');
    
    // Add label if provided
    if (label) {
      const labelDiv = document.createElement('div');
      labelDiv.style.display = 'flex';
      labelDiv.style.alignItems = 'center';
      labelDiv.style.gap = '8px';
      labelDiv.style.marginBottom = '5px';
      
      const labelText = document.createElement('span');
      labelText.textContent = label;
      labelText.style.color = '#ffffff';
      labelText.style.fontFamily = 'monospace';
      labelDiv.appendChild(labelText);
      
      // Add info icon
      const infoIcon = document.createElement('span');
      infoIcon.textContent = 'ⓘ';
      infoIcon.style.cursor = 'pointer';
      infoIcon.style.color = '#888';
      infoIcon.style.fontSize = '14px';
      infoIcon.style.userSelect = 'none';
      infoIcon.style.transition = 'color 0.2s';
      infoIcon.title = 'Click for information';
      
      infoIcon.addEventListener('mouseenter', () => {
        infoIcon.style.color = '#fff';
      });
      infoIcon.addEventListener('mouseleave', () => {
        infoIcon.style.color = '#888';
      });
      
      infoIcon.addEventListener('click', (e) => {
        e.stopPropagation();
        showAnalyzerInfo(label, infoIcon);
      });
      
      labelDiv.appendChild(infoIcon);
      containerDiv.appendChild(labelDiv);
    }

    containerDiv.appendChild(canvas);
    this.container.appendChild(containerDiv);
    this.visualizations.push(viz);
    return viz;
  }

  /**
   * Get the appropriate visualizer function for an analyzer instance
   */
  private getVisualizer(
    analyzer: any,
    canvas: HTMLCanvasElement,
    options?: { width?: number; height?: number; color?: string; backgroundColor?: string }
  ): (data: any) => void {
    if (analyzer instanceof MFCCAnalyzer) {
      return (data: number[]) => visualizeMFCC(canvas, data, options);
    } else if (analyzer instanceof PLPAnalyzer) {
      return (data: number[]) => visualizePLP(canvas, data, options);
    } else if (analyzer instanceof ChromaFeaturesAnalyzer) {
      return (data: number[]) => visualizeChroma(canvas, data, options);
    } else if (analyzer instanceof LPCAnalyzer) {
      return (data: number[]) => visualizeLPC(canvas, data, options);
    } else if (analyzer instanceof ConstantQTransformAnalyzer) {
      return (data: Float32Array) => visualizeCQT(canvas, data, options);
    } else if (analyzer instanceof WaveletTransformAnalyzer) {
      return (data: Float32Array[]) => visualizeWavelet(canvas, data, options);
    } else if (analyzer instanceof WaveformEnvelopeAnalyzer) {
      return (data: Float32Array) => visualizeEnvelope(canvas, data, options);
    } else if (analyzer instanceof AutocorrelationAnalyzer) {
      return (data: Float32Array) => visualizeAutocorrelation(canvas, data, options);
    } else if (analyzer instanceof SpectralCentroidAnalyzer) {
      return (data: number) => visualizeSpectralCentroid(canvas, data, options);
    } else if (analyzer instanceof SpectralRolloffAnalyzer) {
      return (data: number) => visualizeSpectralRolloff(canvas, data, options);
    } else if (analyzer instanceof SpectralBandwidthAnalyzer) {
      return (data: number) => visualizeSpectralBandwidth(canvas, data, options);
    } else if (analyzer instanceof SpectralFlatnessAnalyzer) {
      return (data: number) => visualizeSpectralFlatness(canvas, data, options);
    } else if (analyzer instanceof ZeroCrossingRateAnalyzer) {
      return (data: number) => visualizeZeroCrossingRate(canvas, data, options);
    } else if (analyzer instanceof RMSEAnalyzer) {
      return (data: number) => visualizeRMSE(canvas, data, options);
    } else if (analyzer instanceof MelSpectrogramAnalyzer) {
      return (data: number[]) => visualizeMelSpectrogram(canvas, data, options);
    } else if (analyzer instanceof FFTAnalyzer) {
      return (data: Float32Array) => visualizeFFT(canvas, data, options);
    } else {
      throw new Error(`Unknown analyzer type: ${analyzer.constructor.name}`);
    }
  }

  /**
   * Update all visualizations with new PCM data
   */
  update(pcm: Float32Array) {
    for (const viz of this.visualizations) {
      try {
        const result = viz.analyzer.analyzeFrame(pcm);
        viz.visualize(result);
      } catch (e) {
        // Skip if frame size doesn't match
        console.warn(`Visualization error for ${viz.label || 'unknown'}:`, e);
      }
    }
  }

  /**
   * Clear all visualizations
   */
  clear() {
    for (const viz of this.visualizations) {
      const ctx = viz.canvas.getContext('2d');
      if (ctx) {
        ctx.clearRect(0, 0, viz.canvas.width, viz.canvas.height);
      }
    }
  }

  /**
   * Remove all visualizations
   */
  removeAll() {
    this.visualizations.forEach(viz => {
      viz.canvas.remove();
    });
    this.visualizations = [];
  }
}

/**
 * Show analyzer information tooltip/modal
 */
function showAnalyzerInfo(label: string, triggerElement: HTMLElement) {
  const info = analyzerInfoMap.get(label);
  if (!info) return;
  
  // Remove existing tooltip if any
  const existing = document.getElementById('analyzer-info-tooltip');
  if (existing) {
    existing.remove();
    return;
  }
  
  // Create tooltip
  const tooltip = document.createElement('div');
  tooltip.id = 'analyzer-info-tooltip';
  tooltip.style.position = 'fixed';
  tooltip.style.backgroundColor = '#1a1a1a';
  tooltip.style.border = '1px solid #444';
  tooltip.style.borderRadius = '8px';
  tooltip.style.padding = '16px';
  tooltip.style.maxWidth = '500px';
  tooltip.style.zIndex = '10000';
  tooltip.style.color = '#fff';
  tooltip.style.fontFamily = 'system-ui, sans-serif';
  tooltip.style.fontSize = '14px';
  tooltip.style.lineHeight = '1.6';
  tooltip.style.boxShadow = '0 4px 12px rgba(0,0,0,0.5)';
  
  // Position tooltip near the trigger
  const rect = triggerElement.getBoundingClientRect();
  tooltip.style.left = `${rect.right + 10}px`;
  tooltip.style.top = `${rect.top}px`;
  
  // Adjust if off-screen
  setTimeout(() => {
    const tooltipRect = tooltip.getBoundingClientRect();
    if (tooltipRect.right > window.innerWidth) {
      tooltip.style.left = `${rect.left - tooltipRect.width - 10}px`;
    }
    if (tooltipRect.bottom > window.innerHeight) {
      tooltip.style.top = `${window.innerHeight - tooltipRect.height - 10}px`;
    }
  }, 0);
  
  // Build content
  tooltip.innerHTML = `
    <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 12px;">
      <h3 style="margin: 0; color: #fff; font-size: 18px;">${info.name}</h3>
      <button id="close-tooltip" style="background: none; border: none; color: #888; cursor: pointer; font-size: 20px; padding: 0; width: 24px; height: 24px; line-height: 1;">×</button>
    </div>
    <div style="margin-bottom: 12px;">
      <strong style="color: #aaa;">Description:</strong>
      <p style="margin: 4px 0 0 0;">${info.description}</p>
    </div>
    <div style="margin-bottom: 12px;">
      <strong style="color: #aaa;">Purpose:</strong>
      <p style="margin: 4px 0 0 0;">${info.purpose}</p>
    </div>
    <div style="margin-bottom: 12px;">
      <strong style="color: #aaa;">Computation:</strong>
      <p style="margin: 4px 0 0 0; font-family: monospace; font-size: 12px; color: #ccc;">${info.computation}</p>
    </div>
    <div>
      <strong style="color: #aaa;">Resources:</strong>
      <ul style="margin: 4px 0 0 0; padding-left: 20px;">
        ${info.resources.map(r => 
          `<li style="margin: 4px 0;"><a href="${r.url}" target="_blank" rel="noopener noreferrer" style="color: #4a9eff; text-decoration: none;">${r.title}</a></li>`
        ).join('')}
      </ul>
    </div>
  `;
  
  // Close button handler
  const closeBtn = tooltip.querySelector('#close-tooltip') as HTMLElement;
  closeBtn?.addEventListener('click', () => tooltip.remove());
  closeBtn?.addEventListener('mouseenter', () => {
    closeBtn.style.color = '#fff';
  });
  closeBtn?.addEventListener('mouseleave', () => {
    closeBtn.style.color = '#888';
  });
  
  // Close on click outside
  const closeOnOutsideClick = (e: MouseEvent) => {
    if (!tooltip.contains(e.target as Node) && e.target !== triggerElement) {
      tooltip.remove();
      document.removeEventListener('click', closeOnOutsideClick);
    }
  };
  setTimeout(() => {
    document.addEventListener('click', closeOnOutsideClick);
  }, 0);
  
  document.body.appendChild(tooltip);
}
