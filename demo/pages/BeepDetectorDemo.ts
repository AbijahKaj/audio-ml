/**
 * Voicemail Beep Detector Demo Page
 */

import { VoicemailBeepDetector } from '../../src/applications/detection/VoicemailBeepDetector';
import { AudioInput } from '../components/AudioInput';
import { AudioInputUI } from '../components/AudioInputUI';

const sampleRate = 44100;

export function createBeepDetectorDemo(container: HTMLElement): () => void {
  let beepDetector: VoicemailBeepDetector | null = null;
  let audioInput: AudioInput | null = null;
  let logContainer: HTMLDivElement | null = null;
  let statsContainer: HTMLDivElement | null = null;

  // Create audio input
  audioInput = new AudioInput(sampleRate);
  audioInput.setTargetFrameSize(2048); // Set to match beep detector's FFT size
  new AudioInputUI(container, audioInput);

  // Create beep detector with frequency ranges
  beepDetector = new VoicemailBeepDetector({ 
    sampleRate, 
    fftSize: 2048,
    frequencyRanges: [
      { min: 400, max: 500, name: 'Low beep (400-500 Hz)' },
      { min: 900, max: 1100, name: 'Mid beep (900-1100 Hz)' },
      { min: 1400, max: 1600, name: 'High beep (1400-1600 Hz)' },
      { min: 1900, max: 2100, name: 'Very high beep (1900-2100 Hz)' }
    ]
  });

  // Create stats container
  statsContainer = document.createElement('div');
  statsContainer.className = 'beep-stats-container';
  container.appendChild(statsContainer);

  const statsTitle = document.createElement('h2');
  statsTitle.textContent = 'Beep Detection Statistics';
  statsTitle.className = 'beep-stats-title';
  statsContainer.appendChild(statsTitle);

  const beepCount = document.createElement('div');
  beepCount.id = 'beep-count';
  beepCount.className = 'beep-count';
  beepCount.textContent = 'Beeps detected: 0';
  statsContainer.appendChild(beepCount);

  const toneCount = document.createElement('div');
  toneCount.id = 'tone-count';
  toneCount.className = 'beep-tone-count';
  toneCount.textContent = 'Tones detected: 0';
  statsContainer.appendChild(toneCount);

  // Create event log
  logContainer = document.createElement('div');
  logContainer.className = 'beep-log-container';
  container.appendChild(logContainer);

  const logTitle = document.createElement('h3');
  logTitle.textContent = 'Detection Events';
  logTitle.className = 'beep-log-title';
  logContainer.appendChild(logTitle);

  const logContent = document.createElement('div');
  logContent.id = 'beep-log-content';
  logContent.className = 'beep-log-content';
  logContainer.appendChild(logContent);

  let beepCounter = 0;
  let toneCounter = 0;

  function addLog(message: string, type: 'beep' | 'tone' | 'info' = 'info'): void {
    const logEntry = document.createElement('div');
    logEntry.className = `beep-log-entry ${type}`;
    const timestamp = new Date().toLocaleTimeString();
    logEntry.textContent = `[${timestamp}] ${message}`;
    logContent.appendChild(logEntry);
    logContent.scrollTop = logContent.scrollHeight;
  }

  // Connect audio input to beep detector
  const pcmHandler = (pcm: Float32Array) => {
    if (!beepDetector) return;
    beepDetector.processFrame(pcm);
  };

  audioInput.on('pcm-data', pcmHandler);

  // Beep detector event handlers
  beepDetector.on('beep-detected', (result) => {
    beepCounter++;
    beepCount.textContent = `Beeps detected: ${beepCounter}`;
    addLog(
      `🔔 BEEP: ${result.frequency.toFixed(0)} Hz, ${result.duration.toFixed(2)}s, confidence: ${Math.round(result.confidence * 100)}%`,
      'beep'
    );
  });

  beepDetector.on('tone-start', (data) => {
    addLog(`Tone started: ${data.frequency.toFixed(0)} Hz`, 'tone');
  });

  beepDetector.on('tone-end', (data) => {
    toneCounter++;
    toneCount.textContent = `Tones detected: ${toneCounter}`;
    addLog(`Tone ended: ${data.frequency.toFixed(0)} Hz, duration: ${data.duration.toFixed(2)}s`, 'tone');
  });

  // Cleanup
  return () => {
    audioInput?.off('pcm-data', pcmHandler);
    audioInput?.stop();
    beepDetector?.stop();
    if (statsContainer) statsContainer.remove();
    if (logContainer) logContainer.remove();
  };
}
