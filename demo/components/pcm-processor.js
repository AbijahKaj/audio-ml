/**
 * AudioWorklet Processor for capturing PCM data
 * Runs in AudioWorkletGlobalScope
 */
class PCMProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.isActive = false;
    this.sampleRate = sampleRate; // Get sampleRate from AudioWorkletGlobalScope
    
    // Listen for messages from the main thread
    this.port.onmessage = (event) => {
      if (event.data.type === 'start') {
        this.isActive = true;
      } else if (event.data.type === 'stop') {
        this.isActive = false;
      }
    };
  }

  process(inputs, outputs) {
    // Only process if active
    if (!this.isActive) {
      return true; // Keep processor alive
    }

    const input = inputs[0];
    
    // If there's input data, send it to the main thread
    if (input && input.length > 0 && input[0] && input[0].length > 0) {
      const inputChannel = input[0];
      // Create a copy of the PCM data to send
      const pcmData = new Float32Array(inputChannel);
      
      // Send PCM data to main thread via MessagePort
      this.port.postMessage({
        type: 'pcm-data',
        data: pcmData.buffer,
        sampleRate: this.sampleRate
      }, [pcmData.buffer]); // Transfer ownership for efficiency
    }

    return true; // Keep processor alive
  }
}

registerProcessor('pcm-processor', PCMProcessor);
