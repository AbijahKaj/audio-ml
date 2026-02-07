/**
 * VAD (Voice Activity Detection) Demo Page
 */

import { VAD } from '../../src/applications/speech/VAD';
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

  vad.on('frame', () => {
    // Continuous updates handled in pcmHandler
  });

  // Cleanup
  return () => {
    audioInput?.off('pcm-data', pcmHandler);
    audioInput?.stop();
    vad?.stop();
    if (statusContainer) statusContainer.remove();
    if (logContainer) logContainer.remove();
  };
}
