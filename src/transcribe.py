"""
uv run hf download thewh1teagle/phonikud-experiments \
  --repo-type model \
  --include "comparison/audio/piper-phonikud/*" \
  --local-dir ./audio-data

OR

git clone https://huggingface.co/thewh1teagle/phonikud-experiments

uv run hf download --repo-type model thewh1teagle/whisper-heb-ipa-ct2 --local-dir ./whisper-heb-ipa-ct2
uv run src/transcribe.py ./phonikud-experiments/comparison/audio/ ./transcripts
"""
import argparse
import torch
import json
from faster_whisper import WhisperModel
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import threading




# Thread-local storage for models
thread_local = threading.local()

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "cpu"  # faster-whisper doesn't support MPS, fallback to CPU
    else:
        return "cpu"

def get_model(model_path, device):
    """Get or create a model for the current thread"""
    if not hasattr(thread_local, 'model'):
        thread_local.model = WhisperModel(model_path, device=device, compute_type="int8_float16")
    return thread_local.model

def transcribe_file(wav_file, model_path, device):
    """Worker function to transcribe a single file"""
    try:
        model = get_model(model_path, device)
        segments, info = model.transcribe(str(wav_file))
        # Combine all segments into a single text
        text = " ".join([segment.text for segment in segments])
        return wav_file.stem, text.strip()
    except Exception as e:
        return wav_file.stem, f"ERROR: {e}"


def main():
    parser = argparse.ArgumentParser(description="Batch transcribe wav files from subfolders")
    parser.add_argument("input", type=str,
                        help="Input folder containing subfolders with wav files")
    parser.add_argument("output", type=str,
                        help="Output folder to save transcripts")
    parser.add_argument("--model", type=str, default="./whisper-heb-ipa-ct2",
                        help="Model path for CT2 converted model (default: ./whisper-heb-ipa-ct2)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing transcript files instead of skipping")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel workers for transcription (default: 4)")
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")
    print(f"Using {args.workers} parallel workers")

    input_dir = Path(args.input)
    if not input_dir.exists():
        print(f"Input directory {input_dir} does not exist")
        return

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all subfolders in the input directory
    subfolders = [p for p in input_dir.iterdir() if p.is_dir()]
    
    if not subfolders:
        print(f"No subfolders found in {input_dir}")
        return

    print(f"Found {len(subfolders)} subfolders to process")

    for subfolder in subfolders:
        print(f"\n📁 Processing subfolder: {subfolder.name}")
        
        # Find all wav files in this subfolder
        wav_files = list(subfolder.glob("*.wav"))
        
        if not wav_files:
            print(f"No wav files found in {subfolder.name}")
            continue

        transcripts = {}
        output_file = output_dir / f"{subfolder.name}.json"
        
        if output_file.exists() and not args.overwrite:
            print(f"⏭️ Skipping {subfolder.name}, {output_file} already exists")
            continue

        print(f"Found {len(wav_files)} wav files in {subfolder.name}")

        # Use ThreadPoolExecutor for parallel transcription
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(transcribe_file, wav_file, args.model, device): wav_file 
                for wav_file in wav_files
            }
            
            # Process results with progress bar
            for future in tqdm(future_to_file, desc=f"Transcribing {subfolder.name}", unit="file"):
                filename, text = future.result()
                if text.startswith("ERROR:"):
                    print(f"\nFailed to transcribe {filename}: {text[7:]}")  # Remove "ERROR: " prefix
                    transcripts[filename] = ""
                else:
                    transcripts[filename] = text

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(transcripts, f, ensure_ascii=False, indent=2, sort_keys=True)

        print(f"✅ Transcriptions for {subfolder.name} saved to {output_file}")


if __name__ == "__main__":
    main()
