from glob import glob
import pandas as pd
import os
import json
import argparse
from matplotlib import pyplot as plt

def fn2name(fn):
    return os.path.splitext(os.path.basename(fn))[0]

def main():
    parser = argparse.ArgumentParser(description='Create scatter plot of WER vs CER from transcript evaluation results')
    parser.add_argument('input_dir', help='Directory containing JSON transcript evaluation files')
    args = parser.parse_args()
    
    # Load data from JSON files
    json_pattern = os.path.join(args.input_dir, '*json')
    data = {fn2name(fn): json.load(open(fn)) for fn in glob(json_pattern)}
    
    # Create DataFrame
    df = pd.DataFrame([
        {
            'model': k,
            **v['summary']
        }
        for k, v in data.items()
    ]).set_index('model')
    
    # Add color coding
    df['is_ours'] = df.index.str.contains('phonikud') | df.index.str.contains('mock')
    
    # Create single clean matplotlib plot
    plt.figure(figsize=(16, 10))
    
    # Plot our models and other models separately for legend
    our_models = df[df['is_ours']]
    other_models = df[~df['is_ours']]
    
    plt.scatter(other_models['mean_wer'], other_models['mean_cer'], 
               c='blue', alpha=0.7, s=120, label='Other models', edgecolors='black', linewidth=0.5)
    plt.scatter(our_models['mean_wer'], our_models['mean_cer'], 
               c='red', alpha=0.7, s=120, label='Our models', edgecolors='black', linewidth=0.5)
    
    # Add model name labels with manual offset and arrows
    import numpy as np
    
    for i, (idx, row) in enumerate(df.iterrows()):
        # Create a circle of positions around each point
        angle = (i * 2 * np.pi) / len(df)  # Distribute evenly in circle
        radius = 0.25  # Even farther distance from point
        
        offset_x = radius * np.cos(angle)
        offset_y = radius * np.sin(angle)
        
        # Position label away from point
        label_x = row['mean_wer'] + offset_x
        label_y = row['mean_cer'] + offset_y
        
        # Color border based on model type
        border_color = 'red' if row['is_ours'] else 'blue'
        
        # Add annotation with very subtle arrows
        plt.annotate(idx, 
                    xy=(row['mean_wer'], row['mean_cer']),  # Point to connect to
                    xytext=(label_x, label_y),              # Label position
                    fontsize=12, 
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, edgecolor=border_color, linewidth=1.2),
                    ha='center',
                    arrowprops=dict(
                        arrowstyle='-', 
                        color='lightgray', 
                        alpha=0.5, 
                        lw=0.8,
                        shrinkA=15,  # Space from the point
                        shrinkB=8    # Space from the text box
                    ))
    
    plt.xlabel('Mean WER (Word Error Rate)', fontsize=14)
    plt.ylabel('Mean CER (Character Error Rate)', fontsize=14) 
    plt.title('TTS Model Performance: WER vs CER', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12, loc='upper right')
    plt.grid(True, alpha=0.3)
    
    # Add very large padding to axes for labels
    x_range = df['mean_wer'].max() - df['mean_wer'].min()
    y_range = df['mean_cer'].max() - df['mean_cer'].min()
    plt.xlim(df['mean_wer'].min() - x_range*0.5, df['mean_wer'].max() + x_range*0.5)
    plt.ylim(df['mean_cer'].min() - y_range*0.5, df['mean_cer'].max() + y_range*0.5)
    
    plt.tight_layout()
    plt.savefig('image.jpg', dpi=300, bbox_inches='tight')
    print(f"Scatter plot with legend table saved to image.jpg")
    
    # Print summary statistics
    print(f"\nLoaded {len(df)} models from {args.input_dir}")
    print(f"Our models: {df['is_ours'].sum()}")
    print(f"Other models: {(~df['is_ours']).sum()}")

if __name__ == '__main__':
    main()
