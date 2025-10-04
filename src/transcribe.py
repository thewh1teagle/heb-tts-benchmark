"""
Transcribe wav files to IPA transcripts.
The format of the input directoy should be subfolders with wav files (each subfolder is different model)

git clone https://huggingface.co/thewh1teagle/phonikud-experiments
uv run hf download --repo-type model thewh1teagle/whisper-heb-ipa-large-v3-turbo-ct2 --local-dir ./whisper-heb-ipa-large-v3-turbo-ct2
uv run src/transcribe.py ./to_transcribe ./transcripts
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

def get_device(args):
    if args.device:
        return args.device
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "cpu"  # faster-whisper doesn't support MPS, fallback to CPU
    else:
        return "cpu"

def get_model(args):
    """Get or create a model for the current thread"""
    if not hasattr(thread_local, 'model'):
        thread_local.model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)
    return thread_local.model

def get_input_folders(input_dir):
    input_path = Path(input_dir)
    
    # If input has wav files directly, use it
    if list(input_path.glob("*.wav")):
        return [(input_path, input_path.name)]
    
    # Otherwise, find subfolders with wav files
    folders = []
    for subfolder in input_path.iterdir():
        if subfolder.is_dir() and list(subfolder.glob("*.wav")):
            folders.append((subfolder, subfolder.name))
    
    return folders

def transcribe_file(wav_file, args):
    """Worker function to transcribe a single file"""
    try:
        model = get_model(args)
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
    parser.add_argument("--model", type=str, default="./whisper-heb-ipa-large-v3-turbo-ct2",
                        help="Model path for CT2 converted model (default: ./whisper-heb-ipa-large-v3-turbo-ct2)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing transcript files instead of skipping")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel workers for transcription (default: 4)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use for transcription (default: auto-detect)")
    parser.add_argument("--compute-type", type=str, default="int8_float16", choices=["int8_float16", "int8_float32"],
                        help="Compute type for transcription (default: int8_float16)")
    args = parser.parse_args()

    args.device = get_device(args)
    print(f"Using device: {args.device}")
    print(f"Using {args.workers} parallel workers")

    input_dir = Path(args.input)
    if not input_dir.exists():
        print(f"Input directory {input_dir} does not exist")
        return

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get input folders to process (handles both direct wav files and subfolders)
    input_folders = get_input_folders(input_dir)
    
    if not input_folders:
        print(f"No wav files found in {input_dir} or its subfolders")
        return

    for folder_path, folder_name in input_folders:
        print(f"\n📁 Processing folder: {folder_name}")
        
        # Find all wav files in this folder
        wav_files = list(folder_path.glob("*.wav"))
        
        if not wav_files:
            print(f"No wav files found in {folder_name}")
            continue

        transcripts = {}
        output_file = output_dir / f"{folder_name}.json"
        
        if output_file.exists() and not args.overwrite:
            print(f"⏭️ Skipping {folder_name}, {output_file} already exists")
            continue

        print(f"Found {len(wav_files)} wav files in {folder_name}")

        # Use ThreadPoolExecutor for parallel transcription
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(transcribe_file, wav_file, args): wav_file 
                for wav_file in wav_files
            }
            
            # Process results with progress bar
            for future in tqdm(future_to_file, desc=f"Transcribing {folder_name}", unit="file"):
                filename, text = future.result()
                if text.startswith("ERROR:"):
                    print(f"\nFailed to transcribe {filename}: {text[7:]}")  # Remove "ERROR: " prefix
                    transcripts[filename] = ""
                else:
                    transcripts[filename] = text

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(transcripts, f, ensure_ascii=False, indent=2, sort_keys=True)

        print(f"✅ Transcriptions for {folder_name} saved to {output_file}")


if __name__ == "__main__":
    main()
