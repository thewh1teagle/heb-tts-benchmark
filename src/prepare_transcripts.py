"""
uv run src/prepare_transcripts.py ./transcripts ./transcripts_cleaned
"""
import json
import argparse
from pathlib import Path
import re
import pandas as pd

def clean_phonemes(text):
    """
    Clean phoneme text by removing punctuation but keeping spaces.
    """
    if not text:
        return ""
    
    # Remove punctuation but keep spaces and letters/phonemes
    cleaned_text = re.sub(r'[^\w\s\u0294\u0261\u0281\u02c8]', '', text)
    
    # Clean up multiple spaces
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
    return cleaned_text


def main():
    parser = argparse.ArgumentParser(description="Clean phoneme transcripts from JSON files")
    parser.add_argument("input", type=str,
                        help="Input folder containing JSON transcript files")
    parser.add_argument("output", type=str,
                        help="Output folder to save cleaned transcripts")
    parser.add_argument("--gold-csv", type=str, default="saspeech_100_gold.csv",
                        help="Path to gold standard CSV file (default: saspeech_100_gold.csv)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing output files")
    args = parser.parse_args()

    input_dir = Path(args.input)
    if not input_dir.exists():
        print(f"Input directory {input_dir} does not exist")
        return

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read the gold standard CSV
    gold_csv_path = Path(args.gold_csv)
    if not gold_csv_path.exists():
        print(f"Gold CSV file {gold_csv_path} does not exist")
        return
    
    print(f"📊 Loading gold standard data from {gold_csv_path}")
    gold_df = pd.read_csv(gold_csv_path)
    
    # Create a dictionary for quick lookup by ID
    gold_data = {}
    for _, row in gold_df.iterrows():
        gold_data[row['id']] = {
            'gt_text': row['text'],
            'gt_phonemes': row['phonemes']
        }
    
    print(f"Loaded {len(gold_data)} gold standard entries")

    # Find all JSON files in the input directory
    json_files = list(input_dir.glob("*.json"))
    
    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return

    print(f"Found {len(json_files)} JSON files to process")

    for json_file in json_files:
        print(f"\n📄 Processing: {json_file.name}")
        
        output_file = output_dir / json_file.name
        
        if output_file.exists() and not args.overwrite:
            print(f"⏭️ Skipping {json_file.name}, output file already exists")
            continue

        try:
            # Read the input JSON file
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Process each key-value pair
            cleaned_data = {}
            matched_count = 0
            
            for key, value in data.items():
                before_clean = str(value) if value else ""
                after_clean = clean_phonemes(before_clean)
                
                entry = {
                    "before_clean": before_clean,
                    "after_clean": after_clean
                }
                
                # Add ground truth data if available
                if key in gold_data:
                    entry["gt_text"] = gold_data[key]["gt_text"]
                    entry["gt_phonemes"] = gold_data[key]["gt_phonemes"]
                    matched_count += 1
                else:
                    entry["gt_text"] = None
                    entry["gt_phonemes"] = None
                
                cleaned_data[key] = entry
            
            # Save the cleaned data
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(cleaned_data, f, ensure_ascii=False, indent=2, sort_keys=True)
            
            print(f"✅ Cleaned {len(cleaned_data)} entries ({matched_count} matched with gold data), saved to {output_file}")
            
        except Exception as e:
            print(f"❌ Failed to process {json_file.name}: {e}")


if __name__ == "__main__":
    main()