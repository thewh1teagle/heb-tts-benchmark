import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List
import re
from math import pi

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
    """Extract comprehensive metrics from all JSON files in the input directory."""
    metrics = {}
    
    json_files = list(input_dir.glob("*.json"))
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract summary metrics
            summary = data.get("summary", {})
            
            # Use filename without extension as model name
            model_name = json_file.stem
            
            metrics[model_name] = {
                "mean_wer": summary.get("mean_wer", 0.0),
                "mean_cer": summary.get("mean_cer", 0.0),
                "median_wer": summary.get("median_wer", 0.0),
                "median_cer": summary.get("median_cer", 0.0),
                "min_wer": summary.get("min_wer", 0.0),
                "min_cer": summary.get("min_cer", 0.0),
                "valid_entries": summary.get("valid_entries", 0),
                "total_words": summary.get("total_words", 0)
            }
            
        except Exception as e:
            print(f"Warning: Failed to process {json_file.name}: {e}")
    
    return metrics


def create_radar_chart(metrics: Dict[str, Dict[str, float]], output_path: Path):
    """Create a radar chart visualization showing multiple performance dimensions."""
    if not metrics:
        print("No metrics data found to visualize")
        return
    
    # Select top 6 models by mean WER for readability
    sorted_models = sorted(metrics.keys(), key=lambda x: metrics[x]["mean_wer"])[:6]
    
    # Define the metrics we want to show (inverted so higher is better)
    radar_metrics = [
        "WER Performance",
        "CER Performance", 
        "WER Consistency",
        "CER Consistency",
        "Best WER",
        "Best CER"
    ]
    
    # Prepare data - invert metrics so higher values are better
    radar_data = {}
    for model in sorted_models:
        # Invert error rates (1 - error_rate) and scale for better visualization
        wer_perf = max(0, 1 - metrics[model]["mean_wer"])
        cer_perf = max(0, 1 - metrics[model]["mean_cer"])
        wer_consistency = max(0, 1 - (metrics[model]["mean_wer"] - metrics[model]["min_wer"]))
        cer_consistency = max(0, 1 - (metrics[model]["mean_cer"] - metrics[model]["min_cer"]))
        best_wer = max(0, 1 - metrics[model]["min_wer"])
        best_cer = max(0, 1 - metrics[model]["min_cer"])
        
        radar_data[model] = [wer_perf, cer_perf, wer_consistency, cer_consistency, best_wer, best_cer]
    
    # Set up the radar chart
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))
    fig.patch.set_facecolor(BACKGROUND_COLOR)
    
    # Calculate angles for each metric
    angles = [n / float(len(radar_metrics)) * 2 * pi for n in range(len(radar_metrics))]
    angles += angles[:1]  # Complete the circle
    
    # Plot each model
    colors = []
    for i, model in enumerate(sorted_models):
        color = OUR_MODEL_COLOR if is_our_model(model) else OTHER_MODEL_COLOR
        colors.append(color)
        
        values = radar_data[model]
        values += values[:1]  # Complete the circle
        
        ax.plot(angles, values, 'o-', linewidth=2, label=model, color=color, alpha=0.8)
        ax.fill(angles, values, alpha=0.15, color=color)
    
    # Customize the radar chart
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_metrics, fontsize=11, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Add title
    plt.title('TTS Model Performance Radar Chart\n(Higher values indicate better performance)', 
              size=16, fontweight='bold', pad=30, color='#2C3E50')
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=10)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=BACKGROUND_COLOR)
    plt.close()
    
    print(f"✅ Radar chart visualization saved to {output_path}")


def create_heatmap_visualization(metrics: Dict[str, Dict[str, float]], output_path: Path):
    """Create a heatmap showing all metrics for all models."""
    if not metrics:
        print("No metrics data found to visualize")
        return
    
    # Prepare data for heatmap
    models = list(metrics.keys())
    metric_names = ["Mean WER", "Mean CER", "Median WER", "Median CER", "Min WER", "Min CER"]
    
    # Create matrix
    data_matrix = []
    for model in models:
        row = [
            metrics[model]["mean_wer"],
            metrics[model]["mean_cer"],
            metrics[model]["median_wer"],
            metrics[model]["median_cer"],
            metrics[model]["min_wer"],
            metrics[model]["min_cer"]
        ]
        data_matrix.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(data_matrix, index=models, columns=metric_names)
    
    # Sort by Mean WER
    df = df.sort_values('Mean WER')
    
    # Create the heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor(BACKGROUND_COLOR)
    
    # Create heatmap with custom colormap (lower values are better, so use reversed colormap)
    sns.heatmap(df, annot=True, fmt='.3f', cmap='RdYlGn_r', 
                cbar_kws={'label': 'Error Rate'}, ax=ax,
                linewidths=0.5, linecolor='white')
    
    # Customize
    ax.set_title('TTS Model Performance Heatmap\n(Lower values indicate better performance)', 
                 fontsize=14, fontweight='bold', pad=20, color='#2C3E50')
    ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
    ax.set_ylabel('Models', fontsize=12, fontweight='bold')
    
    # Color model names based on type
    y_labels = ax.get_yticklabels()
    for i, label in enumerate(y_labels):
        model_name = label.get_text()
        if is_our_model(model_name):
            label.set_color(OUR_MODEL_COLOR)
            label.set_fontweight('bold')
        else:
            label.set_color(OTHER_MODEL_COLOR)
    
    plt.tight_layout()
    plt.savefig(output_path.with_suffix('.png').with_name(output_path.stem + '_heatmap.jpg'), 
                dpi=300, bbox_inches='tight', facecolor=BACKGROUND_COLOR)
    plt.close()
    
    print(f"✅ Heatmap visualization saved to {output_path.with_name(output_path.stem + '_heatmap.jpg')}")


def create_scatter_plot(metrics: Dict[str, Dict[str, float]], output_path: Path):
    """Create a simple scatter plot of WER vs CER with numbered points and legend."""
    if not metrics:
        print("No metrics data found to visualize")
        return
    
    # Prepare data
    models = list(metrics.keys())
    wer_values = [metrics[model]["mean_wer"] for model in models]
    cer_values = [metrics[model]["mean_cer"] for model in models]
    colors = [OUR_MODEL_COLOR if is_our_model(model) else OTHER_MODEL_COLOR for model in models]
    
    # Create scatter plot with appropriate size for 18 labels
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), gridspec_kw={'width_ratios': [3, 1]})
    fig.patch.set_facecolor(BACKGROUND_COLOR)
    
    # Main scatter plot
    for i, model in enumerate(models):
        ax1.scatter(wer_values[i], cer_values[i], 
                   c=colors[i], s=200, alpha=0.8, edgecolors='white', linewidth=2)
        
        # Add number labels on points
        ax1.annotate(str(i+1), (wer_values[i], cer_values[i]), 
                    ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    
    # Customize main plot
    ax1.set_xlabel('Mean WER (Word Error Rate)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Mean CER (Character Error Rate)', fontsize=14, fontweight='bold')
    ax1.set_title('TTS Model Performance: WER vs CER\n(Lower values indicate better performance)', 
                 fontsize=16, fontweight='bold', pad=20, color='#2C3E50')
    
    # Add grid
    ax1.grid(True, alpha=0.3)
    ax1.set_facecolor('white')
    
    # Add diagonal reference line
    min_val = min(min(wer_values), min(cer_values))
    max_val = max(max(wer_values), max(cer_values))
    ax1.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.4, 
            linewidth=2, label='Perfect WER=CER correlation')
    
    # Create legend table on the right
    ax2.axis('off')
    ax2.set_title('Model Legend', fontsize=14, fontweight='bold', pad=20, color='#2C3E50')
    
    # Create table data
    table_data = []
    for i, model in enumerate(models):
        # Shorten model names for the table
        display_name = model if len(model) <= 25 else model[:22] + "..."
        model_type = "Our Model" if is_our_model(model) else "Other"
        color = OUR_MODEL_COLOR if is_our_model(model) else OTHER_MODEL_COLOR
        
        table_data.append([str(i+1), display_name, f"{wer_values[i]:.3f}", f"{cer_values[i]:.3f}", model_type])
    
    # Create table
    table = ax2.table(cellText=table_data,
                     colLabels=['#', 'Model Name', 'WER', 'CER', 'Type'],
                     cellLoc='left',
                     loc='center',
                     colWidths=[0.08, 0.5, 0.12, 0.12, 0.18])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    
    # Color code the rows
    for i, model in enumerate(models):
        color = OUR_MODEL_COLOR if is_our_model(model) else OTHER_MODEL_COLOR
        # Color the number cell
        table[(i+1, 0)].set_facecolor(color)
        table[(i+1, 0)].set_text_props(weight='bold', color='white')
        # Color the type cell
        table[(i+1, 4)].set_facecolor(color)
        table[(i+1, 4)].set_text_props(weight='bold', color='white')
    
    # Style header
    for j in range(5):
        table[(0, j)].set_facecolor('#34495E')
        table[(0, j)].set_text_props(weight='bold', color='white')
    
    # Add color legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=OUR_MODEL_COLOR, label='Our Models', alpha=0.8),
        Patch(facecolor=OTHER_MODEL_COLOR, label='Other Models', alpha=0.8)
    ]
    ax1.legend(handles=legend_elements, loc='upper right', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path.with_name(output_path.stem + '_scatter.jpg'), 
                dpi=300, bbox_inches='tight', facecolor=BACKGROUND_COLOR)
    plt.close()
    
    print(f"✅ Simplified scatter plot with legend table saved to {output_path.with_name(output_path.stem + '_scatter.jpg')}")


def main():
    parser = argparse.ArgumentParser(description="Create alternative visualizations of WER and CER metrics")
    parser.add_argument("input", type=str,
                        help="Input folder containing JSON files with evaluation results")
    parser.add_argument("--output", type=str, default="image.jpg",
                        help="Output image filename (default: image.jpg)")
    parser.add_argument("--type", type=str, choices=['radar', 'heatmap', 'scatter', 'all'], 
                        default='all', help="Type of visualization to create")
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
    
    print(f"Found metrics for {len(metrics)} models")
    
    # Create visualizations based on type
    if args.type in ['radar', 'all']:
        create_radar_chart(metrics, output_path)
    
    if args.type in ['heatmap', 'all']:
        create_heatmap_visualization(metrics, output_path)
    
    if args.type in ['scatter', 'all']:
        create_scatter_plot(metrics, output_path)
    
    # Print summary
    our_models = [m for m in metrics.keys() if is_our_model(m)]
    other_models = [m for m in metrics.keys() if not is_our_model(m)]
    print(f"📈 Created visualizations for {len(our_models)} our models and {len(other_models)} other models")


if __name__ == "__main__":
    main()
