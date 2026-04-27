#!/usr/bin/env python3
"""
Расчёт рангов моделей по метрикам BLEU, ROUGE-LSum (F1), Levenshtein.

Алгоритм:
  1. Для каждой метрики и каждого датасета — ранжировать модели (дробные ранги при равенстве).
  2. Средний ранг модели по метрике = среднее по датасетам.
  3. Итоговый ранг модели = среднее по трём метрикам.

Использование
─────────────
# Все модели и датасеты, округление до 3 знаков (по умолчанию):
python compute_ranks.py

# Только выбранные модели:
python compute_ranks.py --models trafilatura readability resiliparse

# Только выбранные датасеты:
python compute_ranks.py --datasets canola cleaneval dragnet

# Комбинирование фильтров + округление:
python compute_ranks.py --models trafilatura boilerpipe --datasets canola cetd --round 4

# Сохранить в другой файл:
python compute_ranks.py --output my_ranks.csv
"""

import argparse
import os
import pandas as pd


# ── Константы ──────────────────────────────────────────────────────────────────

METRICS_PATH = "outputs/metrics-computed"

ALL_DATASETS = [
    "canola", "cetd", "cleaneval", "cleanportaleval", "multipage",
    "dragnet", "google-trends-2017", "l3s-gn1", "readability", "scrapinghub",
]

# metric_key → (filename, value_column, ascending)
#   ascending=True  → меньше = лучше (Levenshtein distance)
#   ascending=False → больше = лучше (BLEU, ROUGE)
METRICS_CONFIG = {
    "bleu": {
        "file": os.path.join(METRICS_PATH, "metrics_table_bleu.csv"),
        "col": None,          # значение само по себе
        "ascending": False,
    },
    "rouge_f1": {
        "file": os.path.join(METRICS_PATH, "metrics_table_rouge_f1.csv"),
        "col": None,
        "ascending": False,
    },
    "levenshtein": {
        "file": os.path.join(METRICS_PATH, "metrics_table_levenshtein.csv"),
        "col": None,
        # В таблице хранится similarity ratio (0..1), а не расстояние — выше = лучше
        "ascending": False,
    },
}


# ── Вспомогательные функции ────────────────────────────────────────────────────

def load_metric_table(filepath: str) -> pd.DataFrame:
    """Загрузить таблицу метрик (строки — модели, столбцы — датасеты)."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Файл не найден: {filepath}")
    return pd.read_csv(filepath, index_col=0)


def fractional_rank_series(series: pd.Series, ascending: bool) -> pd.Series:
    """
    Присвоить дробные ранги для одного датасета (одна колонка таблицы).
    NaN-значения получают NaN-ранг и не влияют на ранги других.
    """
    return series.rank(method="average", ascending=ascending, na_option="keep")


def compute_ranks(
    models: list[str] | None = None,
    datasets: list[str] | None = None,
    round_digits: int = 3,
    output_path: str | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Основная функция расчёта рангов.

    Параметры
    ---------
    models       : список моделей для учёта (None = все доступные)
    datasets     : список датасетов для учёта (None = все доступные)
    round_digits : количество знаков после запятой для округления
    output_path  : путь для сохранения CSV (None = не сохранять)
    verbose      : выводить промежуточную информацию

    Возвращает
    ----------
    DataFrame с колонками [bleu, rouge_f1, levenshtein, overall],
    отсортированный по overall по возрастанию (меньший ранг = лучше).
    """
    target_datasets = datasets if datasets is not None else ALL_DATASETS

    metric_avg_ranks: dict[str, pd.Series] = {}

    for metric_key, cfg in METRICS_CONFIG.items():
        table = load_metric_table(cfg["file"])

        # ── Фильтр датасетов ──────────────────────────────────────────────────
        available_datasets = [d for d in target_datasets if d in table.columns]
        missing = [d for d in target_datasets if d not in table.columns]
        if missing and verbose:
            print(f"[{metric_key}] датасеты не найдены в таблице: {missing}")

        table = table[available_datasets]

        # ── Фильтр моделей ────────────────────────────────────────────────────
        if models is not None:
            available_models = [m for m in models if m in table.index]
            missing_models = [m for m in models if m not in table.index]
            if missing_models and verbose:
                print(f"[{metric_key}] модели не найдены: {missing_models}")
            table = table.loc[available_models]
        else:
            available_models = table.index.tolist()

        if table.empty:
            raise ValueError(f"[{metric_key}] После фильтрации таблица пуста.")

        # ── Шаг 1: ранг внутри каждого датасета ──────────────────────────────
        # Ранжируем только среди отфильтрованных моделей, внутри каждого столбца.
        rank_table = table.apply(
            lambda col: fractional_rank_series(col, cfg["ascending"]),
            axis=0,
        )

        if verbose:
            print(f"\n[{metric_key}] Ранги по датасетам (ascending={cfg['ascending']}):")
            print(rank_table.round(round_digits).to_string())

        # ── Шаг 2: средний ранг по метрике ───────────────────────────────────
        # Среднее только по датасетам, в которых у модели есть значение (skipna).
        metric_avg_ranks[metric_key] = rank_table.mean(axis=1, skipna=True)

    # ── Шаг 3: итоговый ранг ─────────────────────────────────────────────────
    result = pd.DataFrame(metric_avg_ranks)
    result["overall"] = result.mean(axis=1, skipna=True)

    # ── Шаг 4: сортировка по итоговому рангу ─────────────────────────────────
    result = result.sort_values("overall")

    result = result.round(round_digits)

    if verbose:
        print("\n\nИтоговая таблица рангов (меньше = лучше):")
        print(result.to_string())

    if output_path:
        result.to_csv(output_path)
        if verbose:
            print(f"\nСохранено в: {output_path}")

    return result


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Расчёт рангов моделей по метрикам BLEU, ROUGE-F1, Levenshtein.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        metavar="MODEL",
        help="Список моделей для учёта (по умолчанию — все).",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        metavar="DATASET",
        help=(
            "Список датасетов для учёта (по умолчанию — все). "
            "Используйте 'all' для явного указания всех датасетов. "
            f"Доступные: {', '.join(ALL_DATASETS)}"
        ),
    )
    parser.add_argument(
        "--round",
        dest="round_digits",
        type=int,
        default=3,
        metavar="N",
        help="Количество знаков после запятой для округления (по умолчанию: 3).",
    )
    parser.add_argument(
        "--output",
        default=None,
        metavar="PATH",
        help="Путь для сохранения результата в CSV (по умолчанию: не сохранять).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Не выводить промежуточную информацию.",
    )

    args = parser.parse_args()

    datasets = None
    if args.datasets is not None:
        datasets = None if args.datasets == ["all"] else args.datasets

    compute_ranks(
        models=args.models,
        datasets=datasets,
        round_digits=args.round_digits,
        output_path=args.output,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
