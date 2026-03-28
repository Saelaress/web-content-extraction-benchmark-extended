
import pandas as pd
import glob
import os

def calculate_and_compare_metrics(metric_type):
    # Load the aggregated table
    aggregated_table_path = f'outputs/metrics-computed/metrics_table_{metric_type}.csv'
    try:
        aggregated_df = pd.read_csv(aggregated_table_path, index_col=0)
    except FileNotFoundError:
        print(f"Error: Aggregated table not found at {aggregated_table_path}")
        return

    print(f"Checking metric type: {metric_type}")
    discrepancies_found = False

    # Get all raw metric files for the current metric type
    raw_metric_files = glob.glob(f'outputs/metrics-computed/{metric_type}/*/*.csv')

    # Group raw files by dataset and method to re-calculate aggregated values
    recalculated_data = {}

    for file_path in raw_metric_files:
        parts = file_path.split(os.sep)
        dataset = parts[-2]
        method_name = parts[-1].replace(f'{metric_type}_', '').replace('.csv', '')

        try:
            raw_df = pd.read_csv(file_path)
            # Assuming raw_df has a column with the metric values, e.g., 'value' or the metric_type itself
            # Let's assume the metric value is in a column named 'value' for now, or the first unnamed column
            if 'value' in raw_df.columns:
                mean_value = raw_df['value'].mean()
            elif raw_df.shape[1] > 0:
                # If no 'value' column, take the mean of the first data column
                mean_value = raw_df.iloc[:, 0].mean()
            else:
                print(f"Warning: No data column found in {file_path}. Skipping.")
                continue

            if dataset not in recalculated_data:
                recalculated_data[dataset] = {}
            recalculated_data[dataset][method_name] = mean_value
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    # Compare with aggregated_df
    for dataset, methods in recalculated_data.items():
        for method, recalculated_mean in methods.items():
            try:
                # Check if the method and dataset exist in the aggregated_df
                if method in aggregated_df.index and dataset in aggregated_df.columns:
                    aggregated_value = aggregated_df.loc[method, dataset]
                    if abs(recalculated_mean - aggregated_value) > 1e-6: # Using a small tolerance for floating point comparison
                        print(f"  Discrepancy found for {metric_type} - Method: {method}, Dataset: {dataset}")
                        print(f"    Recalculated Mean: {recalculated_mean:.4f}, Aggregated Value: {aggregated_value:.4f}")
                        discrepancies_found = True
            except KeyError:
                print(f"  Warning: Method '{method}' or Dataset '{dataset}' not found in aggregated table for {metric_type}.")
                continue
            except Exception as e:
                print(f"  Error comparing {method} in {dataset} for {metric_type}: {e}")

    if not discrepancies_found:
        print(f"No discrepancies found for {metric_type}.")
    print("-" * 50)


if __name__ == "__main__":
    metrics_to_check = ['bleu', 'levenshtein', 'rouge_f1', 'rouge_prec', 'rouge_rec']
    for metric in metrics_to_check:
        calculate_and_compare_metrics(metric)
