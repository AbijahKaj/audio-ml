#!/usr/bin/env python3
"""Compare NeMo preprocessor features vs our JS features."""
import json
import numpy as np
import soundfile as sf
import torch
import nemo.collections.asr as nemo_asr

def main():
    print("=== Feature Comparison ===\n")
    
    model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt_ctc-110m")
    model.eval()
    
    wav_path = "/workspace/test_audio/libri_0.wav"
    audio, sr = sf.read(wav_path, dtype="float32")
    
    print(f"Audio: {len(audio)} samples, {sr}Hz, {len(audio)/sr:.2f}s")
    
    # Get NeMo preprocessor features
    audio_tensor = torch.tensor(audio).unsqueeze(0)
    audio_len = torch.tensor([len(audio)])
    
    with torch.no_grad():
        features, feat_len = model.preprocessor(input_signal=audio_tensor, length=audio_len)
    
    features_np = features.cpu().numpy()
    print(f"\nNeMo preprocessor output: {features_np.shape}")
    print(f"  min={features_np.min():.4f}, max={features_np.max():.4f}, mean={features_np.mean():.4f}, std={features_np.std():.4f}")
    
    # Save for JS comparison
    np.save("/workspace/test_audio/libri_0_nemo_features.npy", features_np)
    
    # Also get encoder output
    with torch.no_grad():
        encoded, enc_len = model.encoder(audio_signal=features, length=feat_len)
    
    encoded_np = encoded.cpu().numpy()
    print(f"\nNeMo encoder output: {encoded_np.shape}")
    print(f"  min={encoded_np.min():.4f}, max={encoded_np.max():.4f}, mean={encoded_np.mean():.4f}, std={encoded_np.std():.4f}")
    
    np.save("/workspace/test_audio/libri_0_nemo_encoded.npy", encoded_np)
    
    # Get the actual mel filter bank from the model
    fb = model.preprocessor.featurizer.fb.cpu().numpy()
    window = model.preprocessor.featurizer.window.cpu().numpy()
    print(f"\nMel filter bank: {fb.shape}")
    print(f"Window: {window.shape}")
    np.save("/workspace/test_audio/nemo_filterbank.npy", fb)
    np.save("/workspace/test_audio/nemo_window.npy", window)
    
    # Show what NeMo preprocessor config actually uses
    print(f"\nNeMo preprocessor settings:")
    print(f"  dither: {model.cfg.preprocessor.dither}")
    print(f"  normalize: {model.cfg.preprocessor.normalize}")
    print(f"  window: {model.cfg.preprocessor.window}")
    print(f"  n_fft: {model.cfg.preprocessor.n_fft}")
    print(f"  features: {model.cfg.preprocessor.features}")
    print(f"  sample_rate: {model.cfg.preprocessor.sample_rate}")
    print(f"  window_size: {model.cfg.preprocessor.window_size}")
    print(f"  window_stride: {model.cfg.preprocessor.window_stride}")
    
    # Save first frame of features for debugging
    print(f"\nFirst frame (feature 0-9):")
    print(f"  NeMo: {features_np[0, 0, :10]}")


if __name__ == "__main__":
    main()
