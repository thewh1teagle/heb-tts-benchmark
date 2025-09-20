"""
Synthesis audio from text using phonikud-tts

Download models first:
wget https://huggingface.co/thewh1teagle/phonikud-tts-checkpoints/resolve/main/shaul.onnx -O tts-model.onnx
wget https://huggingface.co/thewh1teagle/phonikud-tts-checkpoints/resolve/main/model.config.json -O tts-model.config.json

uv run src/prepare_audio_phonikud.py ./phonikud_shaul_saspeech
"""

import argparse
from pathlib import Path
from phonikud_tts import Piper
import soundfile as sf
import pandas as pd
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Generate audio from phonemized text in CSV")
    parser.add_argument("output", type=str, help="Output folder to save audio files")
    parser.add_argument("--model", type=str, help="Model path")
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load CSV with phonemized text
    df = pd.read_csv('saspeech_100_gold_synthesis.csv')
    
    # Initialize Piper TTS
    piper = Piper(args.model, 'tts-model.config.json')
    
    print(f"Processing {len(df)} texts from CSV...")
    
    # Iterate through each row and generate audio
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating audio"):
        phonemes = row['text_phonemized']
        
        # Create audio from phonemes
        samples, sample_rate = piper.create(phonemes, is_phonemes=True, length_scale=1.4)
        
        # Save audio file using id column
        filename = f"{row['id']}.wav"
        output_file = output_dir / filename
        sf.write(output_file, samples, sample_rate)
    
    print(f"✅ Generated {len(df)} audio files in {output_dir}")

if __name__ == "__main__":
    main()