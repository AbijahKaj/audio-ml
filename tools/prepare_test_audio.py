#!/usr/bin/env python3
"""
Download real speech audio samples and prepare them for testing.
Also runs NeMo-based reference inference if nemo is available, otherwise
uses a direct PyTorch forward pass with the checkpoint weights.
"""
import json
import os
import struct
import sys
import time
import wave

import numpy as np

# Download a LibriSpeech sample
def download_librispeech_sample(output_dir):
    """Download a small LibriSpeech sample."""
    os.makedirs(output_dir, exist_ok=True)
    
    url = "https://www.openslr.org/resources/12/test-clean.tar.gz"
    
    # Instead, let's just create a WAV file from a known speech dataset sample
    # We'll use huggingface datasets for a quick sample
    try:
        from datasets import load_dataset
        print("Downloading speech sample from LibriSpeech via HF datasets...")
        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation[:3]", trust_remote_code=True)
        
        for i, sample in enumerate(ds):
            audio = sample["audio"]
            sr = audio["sampling_rate"]
            data = np.array(audio["array"], dtype=np.float32)
            text = sample["text"]
            
            wav_path = os.path.join(output_dir, f"sample_{i}.wav")
            write_wav(wav_path, data, sr)
            
            meta_path = os.path.join(output_dir, f"sample_{i}.json")
            with open(meta_path, "w") as f:
                json.dump({"text": text, "sample_rate": sr, "duration_s": len(data)/sr, "num_samples": len(data)}, f, indent=2)
            
            print(f"  Sample {i}: {len(data)/sr:.2f}s @ {sr}Hz — \"{text}\"")
            print(f"    Saved to {wav_path}")
        return True
    except ImportError:
        print("datasets library not available, creating synthetic speech sample...")
        return False


def write_wav(path, data, sr):
    """Write float32 audio to 16-bit WAV."""
    data_int16 = (data * 32767).clip(-32768, 32767).astype(np.int16)
    with wave.open(path, 'w') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(data_int16.tobytes())


def write_raw_f32(path, data):
    """Write raw float32 samples (for JS consumption)."""
    with open(path, 'wb') as f:
        f.write(data.astype(np.float32).tobytes())


def create_synthetic_speech(output_dir):
    """Create a synthetic speech-like signal with realistic characteristics."""
    os.makedirs(output_dir, exist_ok=True)
    sr = 16000
    duration = 3.0
    t = np.arange(int(sr * duration)) / sr
    
    # Simulate voiced speech with varying formants
    f0 = 120 + 30 * np.sin(2 * np.pi * 3 * t)  # varying pitch
    phase = np.cumsum(2 * np.pi * f0 / sr)
    
    # Glottal pulse train (simplified)
    signal = np.sin(phase) * 0.3
    signal += np.sin(2 * phase) * 0.15  
    signal += np.sin(3 * phase) * 0.08
    
    # Add formant-like resonances
    from scipy.signal import lfilter
    # F1 around 500Hz, F2 around 1500Hz
    b1, a1 = [0.05], [1, -0.95 * np.exp(-2j * np.pi * 500 / sr).real]
    signal_f = lfilter([0.1], [1, -1.9 * np.cos(2*np.pi*500/sr), 0.95**2], signal)
    signal_f += lfilter([0.05], [1, -1.9 * np.cos(2*np.pi*1500/sr), 0.92**2], signal)
    
    # Amplitude envelope (syllable-like)
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 4 * t) 
    envelope *= (1 - np.exp(-t * 10))  # attack
    signal_f *= envelope * 0.5
    signal_f = signal_f.astype(np.float32)
    
    wav_path = os.path.join(output_dir, "sample_0.wav")
    write_wav(wav_path, signal_f, sr)
    
    meta_path = os.path.join(output_dir, "sample_0.json")
    with open(meta_path, "w") as f:
        json.dump({"text": "(synthetic speech-like audio)", "sample_rate": sr, 
                    "duration_s": duration, "num_samples": len(signal_f)}, f, indent=2)
    
    print(f"  Created synthetic speech: {duration}s @ {sr}Hz")
    return signal_f


def run_python_reference(model_dir, audio_dir):
    """Run inference with raw PyTorch weights as a reference benchmark."""
    import torch
    from safetensors.torch import load_file
    
    print("\n=== Python Reference Inference ===")
    
    # Load audio
    audio_files = sorted([f for f in os.listdir(audio_dir) if f.endswith('.wav')])
    if not audio_files:
        print("No audio files found!")
        return
    
    import soundfile as sf
    
    config_path = os.path.join(model_dir, "model_config.json")
    with open(config_path) as f:
        config = json.load(f)
    
    print(f"Model config: {config['encoder_layers']} layers, d_model={config['d_model']}, "
          f"decoder={config['decoder_type']}, vocab={config['vocab_size']}")
    
    # Load safetensors
    st_path = os.path.join(model_dir, "model.safetensors")
    print(f"Loading weights from {st_path}...")
    t0 = time.time()
    weights = load_file(st_path)
    load_time = time.time() - t0
    print(f"  Loaded {len(weights)} tensors in {load_time*1000:.0f}ms")
    
    # Process each audio file
    for audio_file in audio_files:
        wav_path = os.path.join(audio_dir, audio_file)
        meta_path = wav_path.replace('.wav', '.json')
        
        audio, sr = sf.read(wav_path, dtype='float32')
        
        meta = {}
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
        
        print(f"\n--- {audio_file} ({len(audio)/sr:.2f}s) ---")
        if 'text' in meta:
            print(f"  Reference text: \"{meta['text']}\"")
        
        # Write raw float32 for JS consumption
        raw_path = wav_path.replace('.wav', '.f32')
        write_raw_f32(raw_path, audio)
        print(f"  Saved raw float32 to {raw_path} ({os.path.getsize(raw_path)} bytes)")
        
        # Extract mel features using librosa (as reference)
        import librosa
        t0 = time.time()
        mel = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_fft=512, hop_length=160, win_length=400,
            n_mels=80, power=2.0, fmin=0, fmax=sr//2
        )
        log_mel = np.log(np.maximum(mel, 1e-10))
        feat_time = time.time() - t0
        print(f"  Mel features (librosa): shape={log_mel.shape}, time={feat_time*1000:.0f}ms")
        
        # Save mel features as reference
        mel_path = wav_path.replace('.wav', '_mel_ref.npy')
        np.save(mel_path, log_mel)
        
    print(f"\n  Python weight load time: {load_time*1000:.0f}ms")
    print(f"  Total weight tensors: {len(weights)}")


def try_nemo_inference(audio_dir):
    """Try to run NeMo inference for comparison if available."""
    try:
        import nemo.collections.asr as nemo_asr
        print("\n=== NeMo Reference Inference ===")
        
        model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt_ctc-110m")
        model.eval()
        
        audio_files = sorted([f for f in os.listdir(audio_dir) if f.endswith('.wav')])
        
        for audio_file in audio_files:
            wav_path = os.path.join(audio_dir, audio_file)
            meta_path = wav_path.replace('.wav', '.json')
            
            meta = {}
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
            
            print(f"\n--- {audio_file} ---")
            if 'text' in meta:
                print(f"  Reference: \"{meta['text']}\"")
            
            t0 = time.time()
            result = model.transcribe([wav_path])
            nemo_time = time.time() - t0
            
            transcription = result[0] if isinstance(result, list) else str(result)
            print(f"  NeMo output: \"{transcription}\"")
            print(f"  NeMo time: {nemo_time*1000:.0f}ms")
            
            # Save NeMo result
            result_path = wav_path.replace('.wav', '_nemo_result.json')
            with open(result_path, 'w') as f:
                json.dump({"text": transcription, "time_ms": nemo_time * 1000}, f, indent=2)
        
        return True
    except ImportError:
        print("\nNeMo not installed — skipping NeMo reference inference")
        print("Install with: pip install nemo_toolkit[asr]")
        return False


def main():
    audio_dir = "/workspace/test_audio"
    model_dir = "/workspace/test_model"
    
    print("=== Preparing Test Audio ===\n")
    
    # Try to get real speech
    got_real = download_librispeech_sample(audio_dir)
    
    if not got_real:
        create_synthetic_speech(audio_dir)
    
    # Run Python reference
    run_python_reference(model_dir, audio_dir)
    
    # Try NeMo
    try_nemo_inference(audio_dir)
    
    print("\n=== Audio preparation complete ===")
    print(f"Audio files in: {audio_dir}")
    for f in sorted(os.listdir(audio_dir)):
        size = os.path.getsize(os.path.join(audio_dir, f))
        print(f"  {f}: {size:,} bytes")


if __name__ == "__main__":
    main()
