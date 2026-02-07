/**
 * Audio Denoiser Demo Page
 * Records denoised audio output
 */

import { AudioDenoiser } from '../../src/applications/processing/AudioDenoiser';
import { AudioInput } from '../components/AudioInput';
import { AudioInputUI } from '../components/AudioInputUI';

const sampleRate = 44100;
const fftSize = 2048;

export function createAudioDenoiserDemo(container: HTMLElement): () => void {
  let denoiser: AudioDenoiser | null = null;
  let audioInput: AudioInput | null = null;
  let isRecording: boolean = false;

  // Create audio input
  audioInput = new AudioInput(sampleRate, fftSize);
  new AudioInputUI(container, audioInput);

  // Create denoiser
  denoiser = new AudioDenoiser({ sampleRate, fftSize });

  // Create controls container
  const controlsContainer = document.createElement('div');
  controlsContainer.className = 'denoiser-controls-container';
  container.appendChild(controlsContainer);

  const title = document.createElement('h2');
  title.textContent = 'Audio Denoiser';
  title.className = 'denoiser-title';
  controlsContainer.appendChild(title);

  const recordButton = document.createElement('button');
  recordButton.textContent = 'Start Recording Denoised Audio';
  recordButton.id = 'record-denoised';
  recordButton.className = 'denoiser-record-button';
  controlsContainer.appendChild(recordButton);

  const statusLabel = document.createElement('div');
  statusLabel.id = 'denoiser-status';
  statusLabel.className = 'denoiser-status';
  statusLabel.textContent = 'Waiting for noise estimation...';
  controlsContainer.appendChild(statusLabel);

  const snrLabel = document.createElement('div');
  snrLabel.id = 'denoiser-snr';
  snrLabel.className = 'denoiser-snr';
  snrLabel.textContent = 'SNR: -- dB';
  controlsContainer.appendChild(snrLabel);

  const recordingsContainer = document.createElement('div');
  recordingsContainer.id = 'denoised-recordings';
  recordingsContainer.className = 'denoiser-recordings-container';
  container.appendChild(recordingsContainer);

  const recordingsTitle = document.createElement('h3');
  recordingsTitle.textContent = 'Recorded Denoised Audio';
  recordingsTitle.className = 'denoiser-recordings-title';
  recordingsContainer.appendChild(recordingsTitle);

  // Store denoised frames for recording
  const denoisedFrames: Float32Array[] = [];
  let recordingStartTime: number = 0;

  // PCM handler that processes audio and collects denoised frames when recording
  const recordingPcmHandler = async (pcm: Float32Array) => {
    if (!denoiser) return;
    
    const result = denoiser.processFrame(pcm);
    
    // Update status
    if (result.snr > 0) {
      snrLabel.textContent = `SNR: ${result.snr.toFixed(1)} dB | Noise Reduction: ${Math.round(result.noiseReduction * 100)}%`;
    }

    // Collect denoised frames if recording
    if (isRecording) {
      denoisedFrames.push(new Float32Array(result.audio));
    }
  };

  audioInput.on('pcm-data', recordingPcmHandler);

  // Denoiser event handlers
  denoiser.on('noise-estimated', () => {
    statusLabel.textContent = 'Noise estimated. Ready to denoise.';
    statusLabel.className = 'denoiser-status ready';
    recordButton.disabled = false;
  });

  denoiser.on('snr-updated', (data) => {
    snrLabel.textContent = `SNR: ${data.snr.toFixed(1)} dB`;
  });

  // Recording functionality
  recordButton.addEventListener('click', async () => {
    if (!isRecording) {
      // Start recording
      denoisedFrames.length = 0;
      recordingStartTime = Date.now();
      isRecording = true;
      recordButton.textContent = 'Stop Recording';
      statusLabel.textContent = 'Recording denoised audio...';
      statusLabel.className = 'denoiser-status recording';
    } else {
      // Stop recording and create audio file
      isRecording = false;
      recordButton.textContent = 'Start Recording Denoised Audio';
      statusLabel.textContent = 'Processing recording...';
      statusLabel.className = 'denoiser-status';

      try {
        // Create audio context for reconstruction
        const ctx = new AudioContext({ sampleRate });
        
        // Calculate total length (accounting for potential frame size differences)
        const totalLength = denoisedFrames.reduce((sum, frame) => sum + frame.length, 0);
        
        if (totalLength === 0) {
          statusLabel.textContent = 'No audio recorded.';
          return;
        }

        // Create AudioBuffer and copy frames
        const audioBuffer = ctx.createBuffer(1, totalLength, ctx.sampleRate);
        const channelData = audioBuffer.getChannelData(0);
        let offset = 0;
        
        for (const frame of denoisedFrames) {
          channelData.set(frame, offset);
          offset += frame.length;
        }

        // Convert to WAV
        const wav = audioBufferToWav(audioBuffer);
        const blob = new Blob([wav], { type: 'audio/wav' });
        const url = URL.createObjectURL(blob);
        
        // Create audio player
        const audio = document.createElement('audio');
        audio.controls = true;
        audio.src = url;
        audio.className = 'denoiser-recording-audio';
        
        // Create download link
        const downloadLink = document.createElement('a');
        downloadLink.href = url;
        downloadLink.download = `denoised-${Date.now()}.wav`;
        downloadLink.textContent = 'Download Denoised Audio';
        downloadLink.className = 'denoiser-download-link';
        
        // Create recording entry
        const recordingDiv = document.createElement('div');
        recordingDiv.className = 'denoiser-recording-entry';
        
        const recordingInfo = document.createElement('div');
        recordingInfo.className = 'denoiser-recording-info';
        recordingInfo.textContent = `Duration: ${(totalLength / ctx.sampleRate).toFixed(2)}s | ${denoisedFrames.length} frames`;
        recordingDiv.appendChild(recordingInfo);
        recordingDiv.appendChild(audio);
        recordingDiv.appendChild(downloadLink);
        
        recordingsContainer.appendChild(recordingDiv);

        statusLabel.textContent = 'Recording saved!';
        statusLabel.className = 'denoiser-status ready';

        await ctx.close();
      } catch (error) {
        console.error('Error processing recording:', error);
        statusLabel.textContent = `Error: ${error instanceof Error ? error.message : String(error)}`;
        statusLabel.className = 'denoiser-status error';
      }
    }
  });

  recordButton.disabled = true;

  // Cleanup
  return () => {
    audioInput?.off('pcm-data', recordingPcmHandler);
    audioInput?.stop();
    denoiser?.stop();
  };
}

// Helper function to convert AudioBuffer to WAV
function audioBufferToWav(buffer: AudioBuffer): ArrayBuffer {
  const length = buffer.length;
  const numberOfChannels = buffer.numberOfChannels;
  const sampleRate = buffer.sampleRate;
  const arrayBuffer = new ArrayBuffer(44 + length * numberOfChannels * 2);
  const view = new DataView(arrayBuffer);
  const channels: Float32Array[] = [];
  
  for (let i = 0; i < numberOfChannels; i++) {
    channels.push(buffer.getChannelData(i));
  }

  // WAV header
  const writeString = (offset: number, string: string) => {
    for (let i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.charCodeAt(i));
    }
  };

  writeString(0, 'RIFF');
  view.setUint32(4, 36 + length * numberOfChannels * 2, true);
  writeString(8, 'WAVE');
  writeString(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, numberOfChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * numberOfChannels * 2, true);
  view.setUint16(32, numberOfChannels * 2, true);
  view.setUint16(34, 16, true);
  writeString(36, 'data');
  view.setUint32(40, length * numberOfChannels * 2, true);

  // Convert float samples to 16-bit PCM
  let offset = 44;
  for (let i = 0; i < length; i++) {
    for (let channel = 0; channel < numberOfChannels; channel++) {
      const sample = Math.max(-1, Math.min(1, channels[channel][i]));
      view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7FFF, true);
      offset += 2;
    }
  }

  return arrayBuffer;
}
