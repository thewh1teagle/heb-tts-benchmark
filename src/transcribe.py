"""
uv run hf download thewh1teagle/phonikud-experiments \
  --repo-type model \
  --include "comparison/audio/piper-phonikud/*" \
  --local-dir ./audio-data

OR

git clone https://huggingface.co/thewh1teagle/phonikud-experiments

uv run src/transcribe.py --input ./phonikud-experiments/comparison/audio/ --output ./transcripts

git clone https://huggingface.co/thewh1teagle/whisper-heb-ipa
uv run ct2-transformers-converter \
    --model ./whisper-heb-ipa \
    --output_dir ./whisper-heb-ipa-ct2 \
    --quantization int8_float16
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
    parser = argparse.ArgumentParser(description="Batch transcribe wav files from subfolders")
    parser.add_argument("--input", type=str, required=True,
                        help="Input folder containing subfolders with wav files")
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

        for wav_file in tqdm(wav_files, desc=f"Transcribing {subfolder.name}", unit="file"):
            try:
                result = pipe(str(wav_file))
                transcripts[wav_file.stem] = result["text"]  # key = filename without extension
            except Exception as e:
                print(f"\nFailed to transcribe {wav_file}: {e}")
                transcripts[wav_file.stem] = ""

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(transcripts, f, ensure_ascii=False, indent=2)

        print(f"✅ Transcriptions for {subfolder.name} saved to {output_file}")


if __name__ == "__main__":
    main()
