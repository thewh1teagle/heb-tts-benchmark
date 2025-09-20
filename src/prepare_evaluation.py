
import argparse
import json
from pathlib import Path
import jiwer
from typing import Dict, Any, Tuple
import statistics


def calculate_metrics(reference: str, hypothesis: str) -> Dict[str, float]:
    """Calculate WER, CER, and other metrics between reference and hypothesis."""
    if not reference or not hypothesis:
        return {
            "wer": 1.0,  # 100% error if either is empty
            "cer": 1.0,
            "word_count": len(reference.split()) if reference else 0,
            "char_count": len(reference) if reference else 0
        }
    
    # Calculate WER (Word Error Rate)
    wer = jiwer.wer(reference, hypothesis)
    
    # Calculate CER (Character Error Rate)  
    cer = jiwer.cer(reference, hypothesis)
    
    # Additional metrics
    word_count = len(reference.split())
    char_count = len(reference)
    
    return {
        "wer": wer,
        "cer": cer,
        "word_count": word_count,
        "char_count": char_count
    }


def calculate_summary_metrics(individual_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate summary statistics from individual results."""
    wer_scores = []
    cer_scores = []
    total_words = 0
    total_chars = 0
    valid_entries = 0
    
    for entry_id, entry_data in individual_results.items():
        if "wer" in entry_data and "cer" in entry_data:
            wer_scores.append(entry_data["wer"])
            cer_scores.append(entry_data["cer"])
            total_words += entry_data.get("word_count", 0)
            total_chars += entry_data.get("char_count", 0)
            valid_entries += 1
    
    if not wer_scores:
        return {
            "mean_wer": 0.0,
            "mean_cer": 0.0,
            "median_wer": 0.0,
            "median_cer": 0.0,
            "total_entries": 0,
            "valid_entries": 0,
            "total_words": 0,
            "total_chars": 0
        }
    
    return {
        "mean_wer": statistics.mean(wer_scores),
        "mean_cer": statistics.mean(cer_scores),
        "median_wer": statistics.median(wer_scores),
        "median_cer": statistics.median(cer_scores),
        "min_wer": min(wer_scores),
        "max_wer": max(wer_scores),
        "min_cer": min(cer_scores),
        "max_cer": max(cer_scores),
        "total_entries": len(individual_results),
        "valid_entries": valid_entries,
        "total_words": total_words,
        "total_chars": total_chars
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate transcription quality using WER and CER metrics")
    parser.add_argument("input", type=str,
                        help="Input folder containing JSON files with transcription results")
    parser.add_argument("output", type=str,
                        help="Output folder to save evaluation results")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing output files")
    args = parser.parse_args()

    input_dir = Path(args.input)
    if not input_dir.exists():
        print(f"Input directory {input_dir} does not exist")
        return

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all JSON files in the input directory
    json_files = list(input_dir.glob("*.json"))
    
    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return

    print(f"Found {len(json_files)} JSON files to evaluate")

    for json_file in json_files:
        print(f"\n📊 Evaluating: {json_file.name}")
        
        output_file = output_dir / json_file.name
        
        if output_file.exists() and not args.overwrite:
            print(f"⏭️ Skipping {json_file.name}, output file already exists")
            continue

        try:
            # Read the input JSON file
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Process each entry
            individual_results = {}
            
            for entry_id, entry_data in data.items():
                # Get the cleaned transcription and ground truth phonemes
                hypothesis = entry_data.get("after_clean", "")
                reference = entry_data.get("gt_phonemes", "")
                
                # Calculate metrics
                metrics = calculate_metrics(reference, hypothesis)
                
                # Create individual result entry
                individual_entry = {
                    **entry_data,  # Include original data
                    **metrics      # Add calculated metrics
                }
                
                individual_results[entry_id] = individual_entry
            
            # Calculate summary metrics
            summary = calculate_summary_metrics(individual_results)
            
            # Create final output structure (summary first, individual second)
            evaluation_result = {}
            evaluation_result["summary"] = summary
            evaluation_result["individual"] = individual_results
            
            # Save the evaluation results (preserve order: summary first, individual second)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(evaluation_result, f, ensure_ascii=False, indent=2, sort_keys=False)
            
            print(f"✅ Evaluated {len(individual_results)} entries")
            print(f"   Mean WER: {summary['mean_wer']:.3f}, Mean CER: {summary['mean_cer']:.3f}")
            print(f"   Results saved to {output_file}")
            
        except Exception as e:
            print(f"❌ Failed to evaluate {json_file.name}: {e}")


if __name__ == "__main__":
    main()