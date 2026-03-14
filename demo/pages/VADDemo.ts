/**
 * VAD (Voice Activity Detection) Demo Page
 */

import { VAD } from 'audio-ml/applications';
import { AudioInput } from '../components/AudioInput';
import { AudioInputUI } from '../components/AudioInputUI';

const sampleRate = 44100;

export function createVADDemo(container: HTMLElement): () => void {
  let vad: VAD | null = null;
  let audioInput: AudioInput | null = null;
  let statusContainer: HTMLDivElement | null = null;
  let logContainer: HTMLDivElement | null = null;

  // Create audio input
  audioInput = new AudioInput(sampleRate);
  new AudioInputUI(container, audioInput);

  // Create VAD with more sensitive speech-end detection
  vad = new VAD({ 
    sampleRate, 
    fftSize: 1024,
    silenceFramesRequired: 3, // Reduced from default 5 for faster speech-end detection
    speechFramesRequired: 2    // Reduced from default 3 for faster speech-start detection
  });

  // Create status display
  statusContainer = document.createElement('div');
  statusContainer.id = 'vad-status';
  statusContainer.className = 'vad-status-container';
  container.appendChild(statusContainer);

  const statusTitle = document.createElement('h2');
  statusTitle.textContent = 'Voice Activity Detection';
  statusTitle.className = 'vad-status-title';
  statusContainer.appendChild(statusTitle);

  const currentStatus = document.createElement('div');
  currentStatus.id = 'vad-current-status';
  currentStatus.className = 'vad-current-status waiting';
  currentStatus.textContent = 'Waiting for audio...';
  statusContainer.appendChild(currentStatus);

  const confidenceLabel = document.createElement('div');
  confidenceLabel.id = 'vad-confidence';
  confidenceLabel.className = 'vad-confidence';
  confidenceLabel.textContent = 'Confidence: 0%';
  statusContainer.appendChild(confidenceLabel);

  // Create event log
  logContainer = document.createElement('div');
  logContainer.id = 'vad-log';
  logContainer.className = 'vad-log-container';
  container.appendChild(logContainer);

  const logTitle = document.createElement('h3');
  logTitle.textContent = 'Event Log';
  logTitle.className = 'vad-log-title';
  logContainer.appendChild(logTitle);

  const logContent = document.createElement('div');
  logContent.id = 'vad-log-content';
  logContent.className = 'vad-log-content';
  logContainer.appendChild(logContent);

  function addLog(message: string, type: 'info' | 'speech' | 'silence' = 'info'): void {
    const logEntry = document.createElement('div');
    logEntry.className = `vad-log-entry ${type}`;
    const timestamp = new Date().toLocaleTimeString();
    logEntry.textContent = `[${timestamp}] ${message}`;
    logContent.appendChild(logEntry);
    logContent.scrollTop = logContent.scrollHeight;
  }

  // Connect audio input to VAD
  const pcmHandler = (pcm: Float32Array) => {
    if (!vad) return;
    const result = vad.processFrame(pcm);
    
    // Update status display
    if (result.isSpeech) {
      currentStatus.textContent = '🎤 SPEECH DETECTED';
      currentStatus.className = 'vad-current-status speech';
    } else {
      currentStatus.textContent = '🔇 SILENCE';
      currentStatus.className = 'vad-current-status silence';
    }
    
    confidenceLabel.textContent = `Confidence: ${Math.round(result.confidence * 100)}% | RMSE: ${result.features.rmse.toFixed(4)} | ZCR: ${result.features.zcr.toFixed(4)}`;
  };

  audioInput.on('pcm-data', pcmHandler);
  
  // Start VAD when audio input starts
  audioInput.on('start', () => {
    vad?.start();
  });
  
  // Reset VAD when audio input stops
  audioInput.on('stop', () => {
    vad?.reset();
  });

  // VAD event handlers
  vad.on('speech-start', (data) => {
    addLog(`Speech started (confidence: ${Math.round(data.confidence * 100)}%)`, 'speech');
  });

  vad.on('speech-end', (data) => {
    addLog(`Speech ended (confidence: ${Math.round(data.confidence * 100)}%)`, 'silence');
  });

  // Explainer section at bottom
  const explainer = document.createElement('div');
  explainer.className = 'app-explainer';
  explainer.innerHTML = `
    <h2 class="app-explainer-title">How It Works</h2>
    <p class="app-explainer-description">
      The VAD determines in real-time whether the incoming audio contains speech or silence/noise.
      It combines four low-level analyzers from the <code>audio-ml</code> library, each contributing a weighted score:
    </p>
    <div class="app-explainer-analyzers">
      <div class="app-explainer-analyzer">
        <span class="app-explainer-analyzer-name">RMSEAnalyzer</span>
        <span class="app-explainer-analyzer-weight">40%</span>
        <span class="app-explainer-analyzer-role">Measures signal energy &mdash; speech is louder than background noise</span>
      </div>
      <div class="app-explainer-analyzer">
        <span class="app-explainer-analyzer-name">ZeroCrossingRateAnalyzer</span>
        <span class="app-explainer-analyzer-weight">30%</span>
        <span class="app-explainer-analyzer-role">Counts sign changes per frame &mdash; voiced speech has a lower rate than noise</span>
      </div>
      <div class="app-explainer-analyzer">
        <span class="app-explainer-analyzer-name">SpectralFlatnessAnalyzer</span>
        <span class="app-explainer-analyzer-weight">20%</span>
        <span class="app-explainer-analyzer-role">Ratio of geometric to arithmetic mean of the spectrum &mdash; noise is flat, speech has peaks</span>
      </div>
      <div class="app-explainer-analyzer">
        <span class="app-explainer-analyzer-name">SpectralCentroidAnalyzer</span>
        <span class="app-explainer-analyzer-weight">10%</span>
        <span class="app-explainer-analyzer-role">Weighted average frequency &mdash; speech energy concentrates above ~500 Hz</span>
      </div>
    </div>
    <p class="app-explainer-detail">
      A frame is classified as speech when the combined confidence &ge; 50% <em>and</em> energy plus at least one other indicator are active.
      Temporal smoothing prevents flickering: speech requires 2 consecutive frames to trigger, silence requires 3.
    </p>
  `;
  container.appendChild(explainer);

  // Cleanup
  return () => {
    audioInput?.off('pcm-data', pcmHandler);
    audioInput?.stop();
    vad?.stop();
    if (statusContainer) statusContainer.remove();
    if (logContainer) logContainer.remove();
    explainer.remove();
  };
}
