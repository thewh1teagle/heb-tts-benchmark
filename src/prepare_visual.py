import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List
import re

# Constants for our model patterns and colors
OUR_MODEL_PATTERNS = [
    r"phonikud_enhanced.*",
    r"phonikud_.*",
    r"piper-phonikud",
    r"unvocalized_.*",
    r"vocalized.*",
    r"mamre-phonikud",
    r"zonus-phonikud"
]

# Color scheme
OUR_MODEL_COLOR = "#2E86AB"  # Professional blue
OTHER_MODEL_COLOR = "#A23B72"  # Contrasting purple
BACKGROUND_COLOR = "#F8F9FA"  # Light gray background
GRID_COLOR = "#E9ECEF"  # Subtle grid


def is_our_model(model_name: str) -> bool:
    """Check if a model matches our model patterns."""
    for pattern in OUR_MODEL_PATTERNS:
        if re.match(pattern, model_name, re.IGNORECASE):
            return True
    return False


def extract_metrics_from_files(input_dir: Path) -> Dict[str, Dict[str, float]]:
    """Extract mean WER and CER from all JSON files in the input directory."""
    metrics = {}
    
    json_files = list(input_dir.glob("*.json"))
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract summary metrics
            summary = data.get("summary", {})
            mean_wer = summary.get("mean_wer", 0.0)
            mean_cer = summary.get("mean_cer", 0.0)
            
            # Use filename without extension as model name
            model_name = json_file.stem
            
            metrics[model_name] = {
                "mean_wer": mean_wer,
                "mean_cer": mean_cer
            }
            
        except Exception as e:
            print(f"Warning: Failed to process {json_file.name}: {e}")
    
    return metrics


def create_visualization(metrics: Dict[str, Dict[str, float]], output_path: Path):
    """Create a beautiful grouped bar chart visualization of WER and CER metrics."""
    if not metrics:
        print("No metrics data found to visualize")
        return
    
    # Sort models by WER performance (best first)
    sorted_models = sorted(metrics.keys(), key=lambda x: metrics[x]["mean_wer"])
    
    # Prepare data for plotting
    wer_values = [metrics[model]["mean_wer"] for model in sorted_models]
    cer_values = [metrics[model]["mean_cer"] for model in sorted_models]
    
    # Create DataFrame for easier plotting with color information
    df_data = []
    for model in sorted_models:
        model_type = "Our Models" if is_our_model(model) else "Other Models"
        df_data.extend([
            {"Model": model, "Metric": "WER", "Value": metrics[model]["mean_wer"], "Type": model_type},
            {"Model": model, "Metric": "CER", "Value": metrics[model]["mean_cer"], "Type": model_type}
        ])
    
    df = pd.DataFrame(df_data)
    
    # Set up the plot style
    plt.style.use('default')
    sns.set_style("whitegrid")
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor(BACKGROUND_COLOR)
    
    # Create custom color palette
    palette = {
        ("Our Models", "WER"): OUR_MODEL_COLOR,
        ("Our Models", "CER"): OUR_MODEL_COLOR,
        ("Other Models", "WER"): OTHER_MODEL_COLOR,
        ("Other Models", "CER"): OTHER_MODEL_COLOR
    }
    
    # Create the grouped bar chart
    sns.barplot(data=df, x='Model', y='Value', hue='Metric', ax=ax, 
                palette=[OUR_MODEL_COLOR, OTHER_MODEL_COLOR], alpha=0.8)
    
    # Customize the plot
    ax.set_title('TTS Model Performance Comparison: WER vs CER\nLower values indicate better performance', 
                 fontsize=16, fontweight='bold', pad=20, color='#2C3E50')
    ax.set_xlabel('Models', fontsize=12, fontweight='bold', color='#2C3E50')
    ax.set_ylabel('Error Rate', fontsize=12, fontweight='bold', color='#2C3E50')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right', fontsize=10)
    
    # Color the bars based on model type
    for i, (container, metric) in enumerate(zip(ax.containers, ['WER', 'CER'])):
        for j, (bar, model) in enumerate(zip(container, sorted_models)):
            if is_our_model(model):
                bar.set_color(OUR_MODEL_COLOR)
            else:
                bar.set_color(OTHER_MODEL_COLOR)
            bar.set_edgecolor('white')
            bar.set_linewidth(1.5)
    
    # Add value labels on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=3, fontsize=8, fontweight='bold')
    
    # Customize the plot appearance
    ax.set_facecolor('white')
    ax.grid(True, alpha=0.3, axis='y', color=GRID_COLOR)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#BDC3C7')
    ax.spines['bottom'].set_color('#BDC3C7')
    
    # Update legend to show model types instead of metrics
    handles, labels = ax.get_legend_handles_labels()
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=OUR_MODEL_COLOR, label='Our Models', alpha=0.8),
        Patch(facecolor=OTHER_MODEL_COLOR, label='Other Models', alpha=0.8)
    ]
    
    # Add both legends - one for metrics, one for model types
    metric_legend = ax.legend(handles, labels, title='Metric', loc='upper left', 
                             fontsize=10, title_fontsize=11)
    ax.add_artist(metric_legend)
    
    type_legend = ax.legend(handles=legend_elements, title='Model Type', loc='upper right', 
                           fontsize=10, title_fontsize=11, frameon=True, fancybox=True, shadow=True)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=BACKGROUND_COLOR,
                edgecolor='none')
    plt.close()
    
    print(f"✅ Beautiful visualization saved to {output_path}")
    
    # Print summary
    our_models = [m for m in sorted_models if is_our_model(m)]
    other_models = [m for m in sorted_models if not is_our_model(m)]
    print(f"📊 Visualized {len(our_models)} our models and {len(other_models)} other models")


def main():
    parser = argparse.ArgumentParser(description="Create visualization of WER and CER metrics from evaluation results")
    parser.add_argument("input", type=str,
                        help="Input folder containing JSON files with evaluation results")
    parser.add_argument("--output", type=str, default="image.jpg",
                        help="Output image filename (default: image.jpg)")
    args = parser.parse_args()

    input_dir = Path(args.input)
    if not input_dir.exists():
        print(f"Input directory {input_dir} does not exist")
        return

    output_path = Path(args.output)
    
    print(f"📊 Reading evaluation results from {input_dir}")
    
    # Extract metrics from all JSON files
    metrics = extract_metrics_from_files(input_dir)
    
    if not metrics:
        print("No valid evaluation files found")
        return
    
    print(f"Found metrics for {len(metrics)} models:")
    for model, values in metrics.items():
        print(f"  {model}: WER={values['mean_wer']:.3f}, CER={values['mean_cer']:.3f}")
    
    # Create visualization
    create_visualization(metrics, output_path)


if __name__ == "__main__":
    main()
