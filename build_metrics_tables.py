#!/usr/bin/env python3
"""
Build metrics_table_*.csv from raw per-model-per-dataset CSV files.

Output format: rows = models, columns = datasets, values = mean score.

Usage
─────
# All models and datasets:
python build_metrics_tables.py

# Selected models:
python build_metrics_tables.py --models trafilatura boilerpipe readability

# Selected datasets:
python build_metrics_tables.py --datasets canola cetd dragnet

# Both:
python build_metrics_tables.py --models trafilatura --datasets canola dragnet

# Custom output dir:
python build_metrics_tables.py --output-dir outputs/metrics-computed

poetry run python compute_ranks.py --models trafilatura readability web2text resiliparse bte crawl4ai boilerpipe boilernet dragnet news_please justext newspaper3k go_domdistiller goose3 extractnet bidirectional --datasets canola cetd cleaneval cleanportaleval multipage dragnet google-trends-2017 readability scrapinghub --round 4
"""

import argparse
import os
import glob

import pandas as pd

METRICS_PATH = "outputs/metrics-computed"

ALL_DATASETS = [
    "canola", "cetd", "cleaneval", "cleanportaleval", "multipage",
    "dragnet", "google-trends-2017", "l3s-gn1", "readability", "scrapinghub",
]

# metric_dir → value column(s); for rouge we emit three separate tables
METRIC_CONFIG = {
    "bleu":        {"col": "bleu_score"},
    "levenshtein": {"col": "dist"},
    "rouge_f1":    {"dir": "rouge", "col": "f1"},
    "rouge_prec":  {"dir": "rouge", "col": "prec"},
    "rouge_rec":   {"dir": "rouge", "col": "rec"},
}


def discover_models(metric_dir: str, datasets: list[str]) -> list[str]:
    models = set()
    for ds in datasets:
        pattern = os.path.join(metric_dir, ds, "*.csv")
        for path in glob.glob(pattern):
            fname = os.path.basename(path)
            prefix = os.path.basename(metric_dir) + "_"
            model = fname.removeprefix(prefix).removesuffix(".csv")
            models.add(model)
    return sorted(models)


def build_table(metric_name: str, datasets: list[str], models: list[str], output_dir: str):
    cfg = METRIC_CONFIG[metric_name]
    metric_dir = os.path.join(METRICS_PATH, cfg.get("dir", metric_name))
    value_col = cfg["col"]

    rows: dict[str, dict[str, float]] = {}

    for model in models:
        for ds in datasets:
            subdir = cfg.get("dir", metric_name)
            fname = f"{subdir}_{model}.csv"
            path = os.path.join(metric_dir, ds, fname)
            if not os.path.isfile(path):
                continue

            df = pd.read_csv(path)
            if value_col not in df.columns:
                print(f"  Warning: column '{value_col}' not in {path}, skipping")
                continue

            mean_val = round(df[value_col].mean(), 4)
            rows.setdefault(model, {})[ds] = mean_val

    if not rows:
        print(f"  No data found for {metric_name}, skipping.")
        return

    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = "model"
    # Keep column order consistent with ALL_DATASETS
    ordered_cols = [d for d in ALL_DATASETS if d in table.columns]
    table = table[ordered_cols]

    out_path = os.path.join(output_dir, f"metrics_table_{metric_name}.csv")
    table.to_csv(out_path)
    print(f"  Written: {out_path}  ({len(table)} models × {len(table.columns)} datasets)")


def main():
    parser = argparse.ArgumentParser(description="Build metrics_table_*.csv from raw metric files.")
    parser.add_argument("--datasets", nargs="+", metavar="DS",
                        help="Datasets to include (default: all)")
    parser.add_argument("--models", nargs="+", metavar="MODEL",
                        help="Models to include (default: auto-discover from files)")
    parser.add_argument("--metrics", nargs="+",
                        choices=list(METRIC_CONFIG),
                        metavar="METRIC",
                        help="Metrics to build (default: all). Choices: " + ", ".join(METRIC_CONFIG))
    parser.add_argument("--output-dir", default=METRICS_PATH,
                        help=f"Output directory (default: {METRICS_PATH})")
    args = parser.parse_args()

    datasets = args.datasets if args.datasets else ALL_DATASETS
    metrics  = args.metrics  if args.metrics  else list(METRIC_CONFIG)

    os.makedirs(args.output_dir, exist_ok=True)

    for metric_name in metrics:
        cfg = METRIC_CONFIG[metric_name]
        metric_dir = os.path.join(METRICS_PATH, cfg.get("dir", metric_name))

        if args.models:
            models = args.models
        else:
            models = discover_models(metric_dir, datasets)

        print(f"\nBuilding metrics_table_{metric_name}.csv  ({len(models)} models, {len(datasets)} datasets)")
        build_table(metric_name, datasets, models, args.output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
