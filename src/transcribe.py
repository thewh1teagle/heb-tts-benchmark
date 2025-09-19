"""
uv run hf download thewh1teagle/phonikud-experiments \
  --repo-type model \
  --include "comparison/audio/piper-phonikud/*" \
  --local-dir ./audio-data

uv run src/transcribe.py --input ./audio-data/comparison/audio/piper-phonikud --output ./transcripts
"""
import argparse
import torch
import json
from transformers import pipeline
from pathlib import Path
from tqdm import tqdm


def get_device():
    if torch.cuda.is_available():
        return 0
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def main():
    parser = argparse.ArgumentParser(description="Batch transcribe wav files")
    parser.add_argument("--input", type=str, nargs="+", required=True,
                        help="Input folder(s) containing wav files")
    parser.add_argument("--output", type=str, required=True,
                        help="Output folder to save transcripts")
    parser.add_argument("--model", type=str, default="thewh1teagle/whisper-heb-ipa",
                        help="Model name or path (default: thewh1teagle/whisper-heb-ipa)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing transcript files instead of skipping")
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    pipe = pipeline(
        task="automatic-speech-recognition",
        model=args.model,
        chunk_length_s=30,
        device=device,
    )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    for folder in args.input:
        folder_path = Path(folder)
        if not folder_path.exists():
            print(f"Skipping {folder} (does not exist)")
            continue

        wav_files = list(folder_path.rglob("*.wav"))
        transcripts = {}

        output_file = output_dir / f"{folder_path.name}.json"
        if output_file.exists() and not args.overwrite:
            print(f"⏭️ Skipping {folder_path.name}, {output_file} already exists")
            continue

        for wav_file in tqdm(wav_files, desc=f"Transcribing {folder_path.name}", unit="file"):
            try:
                result = pipe(str(wav_file))
                transcripts[wav_file.stem] = result["text"]  # key = filename without extension
            except Exception as e:
                print(f"\nFailed to transcribe {wav_file}: {e}")
                transcripts[wav_file.stem] = ""

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(transcripts, f, ensure_ascii=False, indent=2)

        print(f"\n✅ Transcriptions for {folder_path.name} saved to {output_file}")


if __name__ == "__main__":
    main()
