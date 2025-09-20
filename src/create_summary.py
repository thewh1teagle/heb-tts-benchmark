from glob import glob
import pandas as pd
import os
import json
import argparse

def fn2name(fn):
    return os.path.splitext(os.path.basename(fn))[0]

def main():
    parser = argparse.ArgumentParser(description='Create summary JSON from transcript evaluation results')
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
    
    # Add model type classification
    df['is_ours'] = df.index.str.contains('phonikud') | df.index.str.contains('mock')
    df['model_type'] = df['is_ours'].map({True: 'Our models', False: 'Other models'})
    
    # Sort by WER for better readability
    df_sorted = df.sort_values('mean_wer')
    
    # Create summary data
    summary_data = {
        'metadata': {
            'total_models': int(len(df)),
            'our_models_count': int(df['is_ours'].sum()),
            'other_models_count': int((~df['is_ours']).sum()),
            'input_directory': args.input_dir
        },
        'models': []
    }
    
    # Add model data
    for i, (model_name, row) in enumerate(df_sorted.iterrows()):
        model_data = {
            'rank': i + 1,
            'model_name': model_name,
            'mean_wer': round(row['mean_wer'], 4),
            'mean_cer': round(row['mean_cer'], 4),
            'model_type': row['model_type'],
            'is_ours': bool(row['is_ours'])
        }
        
        # Add other metrics if available
        for col in df.columns:
            if col not in ['is_ours', 'model_type'] and col.startswith(('mean_', 'std_')):
                model_data[col] = round(float(row[col]), 4)
        
        summary_data['models'].append(model_data)
    
    # Save summary JSON
    with open('summary.json', 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"Summary saved to summary.json")
    print(f"Total models: {len(df)}")
    print(f"Our models: {df['is_ours'].sum()}")
    print(f"Other models: {(~df['is_ours']).sum()}")
    print(f"Best WER: {df_sorted['mean_wer'].iloc[0]:.4f} ({df_sorted.index[0]})")
    print(f"Best CER: {df_sorted['mean_cer'].min():.4f}")

if __name__ == '__main__':
    main()
