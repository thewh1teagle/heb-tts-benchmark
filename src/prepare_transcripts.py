import jiwer
import re
import pandas as pd
import argparse

STRESS = "\u02c8"  # Stress marker
# Core phoneme sets
VOWELS = "aeiou"
CONSONANTS = "".join([
    # Basic consonants
    "bvdhzχtjklmnsfp",
    
    # Affricates and semi-vowels
    "ts",
    "w",
    
    # Special symbols
    "\u0294",  # ʔ (glottal stop)
    "\u0261",  # ɡ (like g)
    "\u0281",  # ʁ/ʀ (like R)
    "tʃ",      # ch
])
HEBREW_PHONEMES = VOWELS + CONSONANTS