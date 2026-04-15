#!/usr/bin/env python3
"""
Update metrics tables to include cascade model results
"""

import pandas as pd
import os
import glob
from pathlib import Path

METRICS_PATH = 'outputs/metrics-computed'
DATASETS = ['canola', 'cetd', 'cleaneval', 'cleanportaleval', 'data-ml-test',
            'dragnet', 'google-trends-2017', 'l3s-gn1', 'newspaper3k', 'readability', 'scrapinghub']
METRIC_TYPES = {
    'bleu': 'bleu_score',
    'levenshtein': 'dist',
    'rouge_f1': 'f1',
    'rouge_prec': 'prec',
    'rouge_rec': 'rec'
}

def aggregate_cascade_metrics():
    """
    Aggregate cascade model metrics from individual files into summary tables
    """

    print("Aggregating cascade metrics...")

    # For each metric type
    for metric_type, value_col in METRIC_TYPES.items():
        print(f"\nProcessing {metric_type}...")

        # Load existing metrics table
        table_path = os.path.join(METRICS_PATH, f'metrics_table_{metric_type}.csv')
        if not os.path.exists(table_path):
            print(f"  Warning: {table_path} not found, skipping")
            continue

        df = pd.read_csv(table_path, index_col=0)

        # Check if cascade already exists
        if 'cascade' in df.index:
            print(f"  cascade already in metrics_table_{metric_type}.csv, removing old entry")
            df = df.drop('cascade')

        # Compute cascade averages for each dataset
        cascade_scores = {}
        for dataset in DATASETS:
            # Handle rouge metrics specially since they're all in one file
            if metric_type.startswith('rouge_'):
                cascade_file = os.path.join(METRICS_PATH, 'rouge', dataset, 'rouge_cascade.csv')
            else:
                cascade_file = os.path.join(METRICS_PATH, metric_type, dataset, f'{metric_type}_cascade.csv')

            if not os.path.exists(cascade_file):
                print(f"  Warning: cascade file not found for {dataset}: {cascade_file}")
                continue

            try:
                cascade_df = pd.read_csv(cascade_file)

                # Get the appropriate value column
                if value_col in cascade_df.columns:
                    avg_score = cascade_df[value_col].mean()
                    cascade_scores[dataset] = avg_score
                    print(f"  {dataset}: {avg_score:.4f}")
                else:
                    print(f"  Warning: column '{value_col}' not found in {cascade_file}")
                    print(f"  Available columns: {cascade_df.columns.tolist()}")

            except Exception as e:
                print(f"  Error processing {cascade_file}: {e}")
                continue

        # Add cascade row to the dataframe
        if cascade_scores:
            # Create a series for cascade with values for all datasets, NaN for missing ones
            cascade_series = pd.Series(
                [cascade_scores.get(ds, float('nan')) for ds in df.columns],
                index=df.columns,
                name='cascade'
            )

            # Add cascade to the dataframe (rounded to 4 decimal places to match other models)
            df = pd.concat([df, cascade_series.round(4).to_frame().T])

            # Save updated metrics table
            df.to_csv(table_path)
            print(f"  Updated {table_path}")

def compute_average_ranks():
    """
    Compute average ranks across all metrics for each model
    """
    print("\n\nComputing average ranks...")

    metric_files = {
        'bleu': 'metrics_table_bleu.csv',
        'levenshtein': 'metrics_table_levenshtein.csv',
        'rouge_f1': 'metrics_table_rouge_f1.csv',
        'rouge_prec': 'metrics_table_rouge_prec.csv',
        'rouge_rec': 'metrics_table_rouge_rec.csv'
    }

    all_ranks = {}

    # For each metric type, load the table and compute ranks
    for metric_name, filename in metric_files.items():
        filepath = os.path.join(METRICS_PATH, filename)
        if not os.path.exists(filepath):
            print(f"  Warning: {filepath} not found, skipping")
            continue

        df = pd.read_csv(filepath, index_col=0)

        # All metrics: higher is better (levenshtein here is ratio/similarity, not distance)
        # So rank descending: rank 1 = best model
        ranks = df.rank(axis=0, method='average', ascending=False)

        # Compute overall rank for each model (average rank across all datasets)
        overall_ranks = ranks.mean(axis=1)
        all_ranks[metric_name] = overall_ranks

    # Create summary dataframe with ranks for each metric and overall
    ranks_df = pd.DataFrame(all_ranks)
    ranks_df['overall'] = ranks_df.mean(axis=1)
    ranks_df = ranks_df.sort_values('overall')

    # Save ranks file
    ranks_path = os.path.join(METRICS_PATH, 'metrics_average_ranks.csv')
    ranks_df.to_csv(ranks_path, float_format='%.3f')
    print(f"  Updated {ranks_path}")

    # Print summary
    print("\nAverage Ranks (lower is better):")
    print(ranks_df.round(3))

if __name__ == '__main__':
    aggregate_cascade_metrics()
    compute_average_ranks()
    print("\n✓ Metrics updated successfully!")
