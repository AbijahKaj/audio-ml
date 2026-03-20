#!/usr/bin/env python3
"""Run NeMo reference inference and benchmark."""
import json
import os
import time
import torch
import numpy as np

def main():
    audio_dir = "/workspace/test_audio"
    
    print("=== NeMo Reference Benchmark ===\n")
    
    import nemo.collections.asr as nemo_asr
    
    print("Loading model: nvidia/parakeet-tdt_ctc-110m")
    t0 = time.time()
    model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt_ctc-110m")
    model.eval()
    load_time = time.time() - t0
    print(f"  Model loaded in {load_time*1000:.0f}ms\n")
    
    audio_files = sorted([f for f in os.listdir(audio_dir) if f.startswith('libri_') and f.endswith('.wav')])
    
    results = []
    for audio_file in audio_files:
        wav_path = os.path.join(audio_dir, audio_file)
        meta_path = wav_path.replace('.wav', '.json')
        
        meta = {}
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
        
        print(f"--- {audio_file} ({meta.get('duration_s', '?')}s) ---")
        print(f"  Reference: \"{meta.get('text', '?')}\"")
        
        # Warm-up run
        with torch.no_grad():
            _ = model.transcribe([wav_path])
        
        # Timed runs
        times = []
        transcription = None
        for run in range(3):
            t0 = time.time()
            with torch.no_grad():
                result = model.transcribe([wav_path])
            elapsed = time.time() - t0
            times.append(elapsed * 1000)
            if transcription is None:
                r = result[0] if isinstance(result, list) else result
                transcription = r.text if hasattr(r, 'text') else str(r)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        print(f"  NeMo output: \"{transcription}\"")
        print(f"  NeMo time: {avg_time:.0f}ms ± {std_time:.0f}ms (3 runs)")
        print(f"  RTF: {avg_time / (meta.get('duration_s', 1) * 1000):.3f}")
        
        result_data = {
            "file": audio_file,
            "reference": meta.get("text", ""),
            "nemo_output": transcription,
            "duration_s": meta.get("duration_s", 0),
            "nemo_avg_ms": avg_time,
            "nemo_std_ms": std_time,
            "nemo_rtf": avg_time / (meta.get('duration_s', 1) * 1000),
        }
        results.append(result_data)
        
        # Save individual result
        result_path = wav_path.replace('.wav', '_nemo_result.json')
        with open(result_path, 'w') as f:
            json.dump(result_data, f, indent=2)
    
    # Save combined results
    combined_path = os.path.join(audio_dir, "nemo_benchmark.json")
    with open(combined_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n=== Summary ===")
    print(f"Model load: {load_time*1000:.0f}ms")
    for r in results:
        print(f"  {r['file']}: {r['nemo_avg_ms']:.0f}ms (RTF={r['nemo_rtf']:.3f})")


if __name__ == "__main__":
    main()
