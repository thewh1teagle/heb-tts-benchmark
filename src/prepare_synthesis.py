"""
wget https://huggingface.co/thewh1teagle/phonikud-onnx/resolve/b806189/phonikud-1.0.int8.onnx

Prepare text/phoneme for synthesis with TTS

uv run src/prepare_synthesis.py saspeech_100_gold_gt.csv saspeech_100_gold_synthesis.csv   
"""
import re
import pandas as pd
from phonikud_onnx import Phonikud
from phonikud import phonemize
import argparse

parser = argparse.ArgumentParser(description='Prepare text/phoneme for synthesis with TTS')
parser.add_argument('input', type=str, default='saspeech_100_gold_gt.csv', help='Input CSV file')
parser.add_argument('output', type=str, default='saspeech_100_gold_synthesis.csv', help='Output CSV file')
args = parser.parse_args()

HEBREW_PATTERN = r'[\u05D0-\u05EA\s]'

phonikud_model = Phonikud("./phonikud-1.0.int8.onnx")

df = pd.read_csv(args.input)
df['text'] = df['text'].apply(lambda x: ''.join(re.findall(HEBREW_PATTERN, str(x))))
df['text_vocalized'] = df['text'].apply(lambda x: phonikud_model.add_diacritics(x))
df['text_phonemized'] = df['text_vocalized'].apply(lambda x: phonemize(x))
df[['id', 'text', 'text_vocalized', 'text_phonemized']].to_csv(args.output, index=False)
print(f"Saved to {args.output}")