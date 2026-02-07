/**
 * Audio Input UI Component
 * Provides UI for microphone recording and file upload
 */

import { AudioInput, AudioInputMode } from './AudioInput';

export class AudioInputUI {
  private container: HTMLDivElement;
  private audioInput: AudioInput;
  private modeSelector: HTMLDivElement;
  private microphoneButton: HTMLButtonElement;
  private fileInput: HTMLInputElement;
  private fileButton: HTMLButtonElement;
  private statusLabel: HTMLDivElement;
  private currentMode: AudioInputMode = 'microphone';

  constructor(container: HTMLElement, audioInput: AudioInput) {
    this.audioInput = audioInput;
    this.container = document.createElement('div');
    this.container.id = 'audio-input-container';
    this.container.style.cssText = `
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 1rem;
      margin-bottom: 2rem;
      padding: 1.5rem;
      background-color: #1a1a1a;
      border-radius: 8px;
      border: 1px solid #333;
    `;
    container.appendChild(this.container);

    this.createModeSelector();
    this.createControls();
    this.createStatusLabel();
    this.setupEventListeners();
  }

  private createModeSelector(): void {
    this.modeSelector = document.createElement('div');
    this.modeSelector.style.cssText = `
      display: flex;
      gap: 0.5rem;
      margin-bottom: 1rem;
    `;

    const micLabel = document.createElement('label');
    micLabel.style.cssText = `
      display: flex;
      align-items: center;
      gap: 0.5rem;
      cursor: pointer;
      color: #fff;
    `;
    const micRadio = document.createElement('input');
    micRadio.type = 'radio';
    micRadio.name = 'audio-mode';
    micRadio.value = 'microphone';
    micRadio.checked = true;
    micRadio.addEventListener('change', async () => {
      if (micRadio.checked) {
        if (this.audioInput.isRunning()) {
          await this.audioInput.stop();
        }
        this.currentMode = 'microphone';
        this.updateUI();
      }
    });
    micLabel.appendChild(micRadio);
    micLabel.appendChild(document.createTextNode('Microphone'));
    this.modeSelector.appendChild(micLabel);

    const fileLabel = document.createElement('label');
    fileLabel.style.cssText = `
      display: flex;
      align-items: center;
      gap: 0.5rem;
      cursor: pointer;
      color: #fff;
    `;
    const fileRadio = document.createElement('input');
    fileRadio.type = 'radio';
    fileRadio.name = 'audio-mode';
    fileRadio.value = 'file';
    fileRadio.addEventListener('change', async () => {
      if (fileRadio.checked) {
        if (this.audioInput.isRunning()) {
          await this.audioInput.stop();
        }
        this.currentMode = 'file';
        this.updateUI();
      }
    });
    fileLabel.appendChild(fileRadio);
    fileLabel.appendChild(document.createTextNode('File Upload'));
    this.modeSelector.appendChild(fileLabel);

    this.container.appendChild(this.modeSelector);
  }

  private createControls(): void {
    const controlsContainer = document.createElement('div');
    controlsContainer.id = 'audio-controls';
    controlsContainer.style.cssText = `
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 1rem;
      flex-wrap: wrap;
    `;

    // Microphone button
    this.microphoneButton = document.createElement('button');
    this.microphoneButton.textContent = 'Start Recording';
    this.microphoneButton.id = 'mic-button';
    controlsContainer.appendChild(this.microphoneButton);

    // File input (hidden)
    this.fileInput = document.createElement('input');
    this.fileInput.type = 'file';
    this.fileInput.accept = 'audio/*';
    this.fileInput.style.display = 'none';
    this.fileInput.id = 'file-input';

    // File button
    this.fileButton = document.createElement('button');
    this.fileButton.textContent = 'Select Audio File';
    this.fileButton.id = 'file-button';
    controlsContainer.appendChild(this.fileButton);

    this.container.appendChild(controlsContainer);
    this.container.appendChild(this.fileInput);
  }

  private createStatusLabel(): void {
    this.statusLabel = document.createElement('div');
    this.statusLabel.id = 'audio-status';
    this.statusLabel.style.cssText = `
      color: #888;
      font-size: 0.9rem;
      margin-top: 0.5rem;
    `;
    this.statusLabel.textContent = 'Ready';
    this.container.appendChild(this.statusLabel);
  }

  private setupEventListeners(): void {
    this.microphoneButton.addEventListener('click', async () => {
      if (this.audioInput.isRunning()) {
        await this.audioInput.stop();
      } else {
        try {
          await this.audioInput.startMicrophone();
        } catch (error) {
          this.updateStatus(`Error: ${error instanceof Error ? error.message : String(error)}`, 'error');
        }
      }
    });

    this.fileButton.addEventListener('click', async () => {
      if (this.audioInput.isRunning()) {
        await this.audioInput.stop();
      } else {
        this.fileInput.click();
      }
    });

    this.fileInput.addEventListener('change', async (e) => {
      const file = (e.target as HTMLInputElement).files?.[0];
      if (file) {
        try {
          await this.audioInput.loadFile(file);
          this.updateStatus(`Playing: ${file.name}`, 'success');
        } catch (error) {
          this.updateStatus(`Error loading file: ${error instanceof Error ? error.message : String(error)}`, 'error');
        }
      }
    });

    this.audioInput.on('start', () => {
      this.updateStatus(this.currentMode === 'microphone' ? 'Recording...' : 'Playing...', 'active');
      this.updateUI();
    });

    this.audioInput.on('stop', () => {
      this.updateStatus('Stopped', 'inactive');
      this.updateUI();
    });

    this.audioInput.on('error', (error) => {
      this.updateStatus(`Error: ${error.message}`, 'error');
    });
  }

  private updateUI(): void {
    const isRunning = this.audioInput.isRunning();
    
    if (this.currentMode === 'microphone') {
      this.microphoneButton.textContent = isRunning ? 'Stop Recording' : 'Start Recording';
      this.microphoneButton.style.display = 'block';
      this.fileButton.style.display = 'none';
    } else {
      this.microphoneButton.style.display = 'none';
      this.fileButton.textContent = isRunning ? 'Stop Playback' : 'Select Audio File';
      this.fileButton.style.display = 'block';
    }
  }

  private updateStatus(message: string, type: 'active' | 'inactive' | 'error' | 'success' = 'inactive'): void {
    this.statusLabel.textContent = message;
    const colors = {
      active: '#00ff88',
      inactive: '#888',
      error: '#ff4444',
      success: '#00ff88'
    };
    this.statusLabel.style.color = colors[type];
  }

  /**
   * Get the audio input instance
   */
  getAudioInput(): AudioInput {
    return this.audioInput;
  }
}
