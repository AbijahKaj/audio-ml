/**
 * Audio Denoiser Demo Page
 * Records denoised audio output
 */

import { AudioDenoiser } from '../../src/applications/processing/AudioDenoiser';
import { AudioInput } from '../components/AudioInput';
import { AudioInputUI } from '../components/AudioInputUI';
// @ts-expect-error - audiobuffer-to-wav doesn't have type definitions
import { toWav } from 'audiobuffer-to-wav';

const sampleRate = 44100;
const fftSize = 2048;

export function createAudioDenoiserDemo(container: HTMLElement): () => void {
  let denoiser: AudioDenoiser | null = null;
  let audioInput: AudioInput | null = null;
  let isRecording: boolean = false;

  // Create audio input with correct frame size for denoiser
  audioInput = new AudioInput(sampleRate);
  audioInput.setTargetFrameSize(fftSize); // Set to match denoiser's FFT size
  new AudioInputUI(container, audioInput);

  // Create denoiser with more lenient noise estimation
  denoiser = new AudioDenoiser({ 
    sampleRate, 
    fftSize,
    noiseEstimationFrames: 8, // Reduced from 10 for faster estimation
    noiseEstimationThreshold: 0.02 // Slightly higher threshold
  });

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
  statusLabel.textContent = 'Start audio input first, then wait for noise estimation (record 2-3 seconds of silence/noise)...';
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

  // Add instructions
  const instructions = document.createElement('div');
  instructions.className = 'denoiser-status';
  instructions.style.cssText = 'margin-top: 0.5rem; font-size: 0.85rem; line-height: 1.4; color: #aaa;';
  instructions.innerHTML = `
    <strong style="color: #fff;">How to use:</strong><br>
    1. Click "Start Recording" above to start audio input<br>
    2. Record 2-3 seconds of silence/background noise (for noise estimation)<br>
    3. Wait for "Noise estimated!" message<br>
    4. Click "Start Recording Denoised Audio" to record denoised output<br>
    5. Speak or play audio - it will be denoised in real-time<br>
    6. Stop recording to save the denoised audio file
  `;
  controlsContainer.appendChild(instructions);

  // Store denoised frames for recording
  const denoisedFrames: Float32Array[] = [];

  // PCM handler that processes audio and collects denoised frames when recording
  const recordingPcmHandler = async (pcm: Float32Array) => {
    if (!denoiser) return;
    
    // Ensure frame is the right size for denoiser (2048)
    let frame = pcm;
    if (pcm.length !== fftSize) {
      frame = new Float32Array(fftSize);
      if (pcm.length < fftSize) {
        frame.set(pcm, 0);
      } else {
        frame.set(pcm.subarray(0, fftSize), 0);
      }
    }
    
    try {
      const result = denoiser.processFrame(frame);
      
      // Update status
      if (result.snr > 0) {
        snrLabel.textContent = `SNR: ${result.snr.toFixed(1)} dB | Noise Reduction: ${Math.round(result.noiseReduction * 100)}%`;
      }

      // Collect denoised frames if recording AND noise estimation is complete
      // Only collect frames that have been actually denoised (snr > 0 means denoising is active)
      if (isRecording) {
        if (result.snr > 0) {
          // This frame was denoised - collect it
          // Check if audio has actual content (not all zeros or very quiet)
          const maxAmplitude = Math.max(...Array.from(result.audio).map(s => Math.abs(s)));
          if (maxAmplitude > 0.0001) {
            denoisedFrames.push(new Float32Array(result.audio));
          }
        } else {
          // Noise estimation not complete yet - don't collect
          // This shouldn't happen if we check before starting, but just in case
        }
      }
    } catch (error) {
      console.error('Error processing frame in denoiser:', error);
    }
  };

  audioInput.on('pcm-data', recordingPcmHandler);

  // Helper function to process and save recording
  const processRecording = async () => {
    if (!isRecording) return;
    
    // Stop recording first
    isRecording = false;
    recordButton.textContent = 'Start Recording Denoised Audio';
    recordButton.disabled = false; // Re-enable button
    
    if (denoisedFrames.length === 0) {
      statusLabel.textContent = 'No audio recorded. Make sure you spoke/played audio after starting recording.';
      statusLabel.className = 'denoiser-status error';
      return;
    }
    
    statusLabel.textContent = 'Processing recording...';
    statusLabel.className = 'denoiser-status';

    try {
      // Create audio context for reconstruction
      const ctx = new AudioContext({ sampleRate });
      
      // Calculate total length (accounting for potential frame size differences)
      const totalLength = denoisedFrames.reduce((sum, frame) => sum + frame.length, 0);
      
      if (totalLength === 0 || denoisedFrames.length === 0) {
        statusLabel.textContent = `No audio recorded. Collected ${denoisedFrames.length} frames. Make sure you spoke/played audio after starting recording.`;
        statusLabel.className = 'denoiser-status error';
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

      // Normalize audio to prevent clipping and ensure audible output
      let maxAmplitude = 0;
      for (let i = 0; i < channelData.length; i++) {
        maxAmplitude = Math.max(maxAmplitude, Math.abs(channelData[i]));
      }
      
      // Normalize to 80% of max to prevent clipping
      if (maxAmplitude > 0) {
        const normalizationFactor = 0.8 / maxAmplitude;
        for (let i = 0; i < channelData.length; i++) {
          channelData[i] *= normalizationFactor;
        }
      }

      // Convert to WAV using library
      const wav = toWav(audioBuffer);
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
  };

  // Auto-process recording when audio input stops
  audioInput.on('stop', async () => {
    if (isRecording) {
      // Wait a bit to ensure all pending frames are processed
      await new Promise(resolve => setTimeout(resolve, 100));
      await processRecording();
    }
  });

  // Start denoiser when audio input starts
  audioInput.on('start', () => {
    denoiser?.start();
    denoiser?.reset(); // Reset to start fresh noise estimation
    statusLabel.textContent = 'Recording... Stay quiet for 2-3 seconds to estimate background noise.';
    statusLabel.className = 'denoiser-status recording';
  });

  // Denoiser event handlers
  denoiser.on('noise-estimation-progress', (data: { progress: number; frames: number }) => {
    statusLabel.textContent = `Estimating noise... ${Math.round(data.progress)}% (${data.frames}/10 frames)`;
    statusLabel.className = 'denoiser-status recording';
  });

  denoiser.on('noise-estimated', () => {
    statusLabel.textContent = '✅ Noise estimated! You can now record denoised audio.';
    statusLabel.className = 'denoiser-status ready';
    recordButton.disabled = false;
  });

  denoiser.on('snr-updated', (data) => {
    snrLabel.textContent = `SNR: ${data.snr.toFixed(1)} dB`;
  });

  // Recording functionality
  recordButton.addEventListener('click', async () => {
    if (!isRecording) {
      // Check if noise estimation is complete
      if (!denoiser || !(denoiser as any).isNoiseEstimationComplete) {
        statusLabel.textContent = '⚠️ Noise estimation not complete yet. Wait for "Noise estimated!" message.';
        statusLabel.className = 'denoiser-status error';
        return;
      }
      
      // Start recording
      denoisedFrames.length = 0;
      isRecording = true;
      recordButton.textContent = 'Stop Recording';
      statusLabel.textContent = 'Recording denoised audio... Speak or play audio now.';
      statusLabel.className = 'denoiser-status recording';
    } else {
      // Stop recording manually (user clicked button)
      // Stop immediately to prevent further frame collection
      isRecording = false;
      recordButton.textContent = 'Start Recording Denoised Audio';
      recordButton.disabled = false;
      await processRecording();
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

