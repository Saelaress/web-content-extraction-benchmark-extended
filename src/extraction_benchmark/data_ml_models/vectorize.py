# -*- coding: utf-8 -*-
"""
Векторизация: data-ml JSON → плоский вектор признаков.

Основано на дипломном модуле vectorize.py, адаптировано для
встроенного использования в web-content-extraction-benchmark.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Порядок извлечения признаков из data-ml (schema v1)
FEATURE_SPEC = [
    # node
    "node.dom_depth",
    "node.dom_depth_norm",
    "node.tag_name",
    "node.tag_is_a",
    "node.tag_is_div",
    "node.tag_is_p",
    "node.tag_is_heading",
    "node.tag_is_article",
    "node.tag_is_nav",
    "node.tag_is_footer",
    "node.tag_is_header",
    "node.parent_tag",
    "node.grandparent_tag",
    "node.has_parent_article",
    "node.is_excluded_tag",
    "node.num_children",
    # subtree
    "subtree.tag_count",
    "subtree.num_leaves",
    "subtree.text_length_chars",
    "subtree.word_count",
    "subtree.link_count",
    "subtree.link_text_length",
    "subtree.link_text_ratio",
    "subtree.text_without_links_ratio",
    "subtree.words_per_tag",
    "subtree.words_per_leaf",
    "subtree.chars_per_descendant",
    "subtree.links_per_descendant",
    "subtree.children_ratio",
    # text
    "text.has_visible_text",
    "text.is_whitespace_only",
    "text.has_only_links",
    "text.digit_ratio",
    "text.r_punctuation",
    "text.ends_with_punctuation",
    "text.num_lines",
    "text.avg_word_length",
    "text.avg_sentence_length",
    "text.nlp_comma_density",
    # meta
    "meta.has_email",
    "meta.has_microdata_article",
    "meta.image_caption_ratio",
    "meta.list_internal_link_ratio",
    # language
    "language.language_code",
    "language.language_confidence",
    # def31
    "def31.word_ratio",
    "def31.hyperlink_ratio",
    "def31.children_ratio_binary",
    "def31.position_ratio",
]

# Имена категориальных колонок (для CatBoost и sklearn)
CATEGORICAL_COLUMNS = [
    "node__tag_name",
    "node__parent_tag",
    "node__grandparent_tag",
    "language__language_code",
]

# Значение для категорий, не встреченных в train (OOV)
UNKNOWN_CATEGORY = "__unknown__"


def _get_nested(data: dict[str, Any], path: str) -> Any:
    """Получить значение по пути вида 'node.dom_depth' или 'language.language_code'."""
    parts = path.split(".", 1)
    if len(parts) == 1:
        return data.get(path)
    key, rest = parts
    sub = data.get(key)
    if sub is None or not isinstance(sub, dict):
        return None
    return _get_nested(sub, rest)


def _flat_key_to_feature_name(path: str) -> str:
    """'node.tag_name' -> 'node__tag_name' (имена колонок без точки)."""
    return path.replace(".", "__")


def data_ml_to_flat_row(data_ml: dict[str, Any]) -> dict[str, Any]:
    """
    Преобразует один data-ml dict в плоский dict с ключами как в FEATURE_SPEC.

    Числа и bool остаются как есть, строки (категориальные) — как есть.
    Пропуски: числа → 0, bool → False, строка → "".
    """
    row: dict[str, Any] = {}
    categorical_paths = {
        "node.tag_name",
        "node.parent_tag",
        "node.grandparent_tag",
        "language.language_code",
    }
    bool_paths = {
        "node.tag_is_a",
        "node.tag_is_div",
        "node.tag_is_p",
        "node.tag_is_heading",
        "node.tag_is_article",
        "node.tag_is_nav",
        "node.tag_is_footer",
        "node.tag_is_header",
        "node.has_parent_article",
        "node.is_excluded_tag",
        "text.has_visible_text",
        "text.is_whitespace_only",
        "text.has_only_links",
        "text.ends_with_punctuation",
        "meta.has_email",
        "meta.has_microdata_article",
    }
    for path in FEATURE_SPEC:
        val = _get_nested(data_ml, path)
        name = _flat_key_to_feature_name(path)
        if val is None:
            if path in categorical_paths:
                row[name] = ""
            elif path in bool_paths:
                row[name] = False
            else:
                row[name] = 0
        elif isinstance(val, bool):
            row[name] = val
        elif isinstance(val, (int, float)):
            row[name] = val
        elif isinstance(val, str):
            row[name] = val
        else:
            row[name] = 0
    return row


def flat_rows_to_arrays(
    flat_rows: list[dict[str, Any]],
    for_sklearn: bool = False,
    label_encoders: dict[str, Any] | None = None,
) -> tuple[np.ndarray, list[str], dict[str, Any] | None]:
    """
    Преобразует список плоских dict в матрицу X и список имён признаков.

    for_sklearn=True: категориальные кодируются через LabelEncoder.
    for_sklearn=False: категориальные остаются строками (для CatBoost через DataFrame).
    """
    from sklearn.preprocessing import LabelEncoder

    if not flat_rows:
        return (
            np.array([]).reshape(0, len(FEATURE_SPEC)),
            [_flat_key_to_feature_name(p) for p in FEATURE_SPEC],
            label_encoders,
        )

    feature_names = [_flat_key_to_feature_name(p) for p in FEATURE_SPEC]
    cat_names = CATEGORICAL_COLUMNS
    if label_encoders is None:
        label_encoders = {name: LabelEncoder() for name in cat_names}

    columns: dict[str, list[Any]] = {name: [] for name in feature_names}
    for row in flat_rows:
        for name in feature_names:
            columns[name].append(row.get(name))

    X_list: list[np.ndarray] = []
    for name in feature_names:
        col = columns[name]
        if name in cat_names:
            le = label_encoders[name]
            if for_sklearn:
                if not hasattr(le, "classes_") or len(le.classes_) == 0:
                    unique_vals = set(str(c) if c != "" else "" for c in col)
                    le.fit(list(unique_vals) + [UNKNOWN_CATEGORY])
                mapped: list[str] = []
                for c in col:
                    s = str(c) if c != "" else ""
                    mapped.append(s if s in le.classes_ else UNKNOWN_CATEGORY)
                enc = le.transform(mapped)
                X_list.append(enc.astype(np.float64))
            else:
                X_list.append(np.array([str(c) if c != "" else "" for c in col], dtype=object))
        else:
            X_list.append(np.array(col, dtype=np.float64))

    X = np.column_stack(X_list)
    return X, feature_names, label_encoders


def prepare_for_sklearn(
    flat_rows: list[dict[str, Any]],
    label_encoders: dict[str, Any] | None = None,
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    """
    Готовит матрицу X (float) и имена признаков для sklearn (LogReg, RF).
    Категориальные кодируются LabelEncoder.
    Возвращает (X, feature_names, fitted_label_encoders).
    """
    X, feature_names, encoders = flat_rows_to_arrays(
        flat_rows, for_sklearn=True, label_encoders=label_encoders
    )
    return X, feature_names, encoders or {}


# Булевые колонки: при обучении приходят из CSV как 0/1; при инференсе — True/False.
# CatBoost может обрабатывать их по-разному; приводим к 0/1 как при обучении.
_BOOL_COLUMNS = {
    "node__tag_is_a", "node__tag_is_div", "node__tag_is_p", "node__tag_is_heading",
    "node__tag_is_article", "node__tag_is_nav", "node__tag_is_footer", "node__tag_is_header",
    "node__has_parent_article", "node__is_excluded_tag",
    "text__has_visible_text", "text__is_whitespace_only", "text__has_only_links",
    "text__ends_with_punctuation", "meta__has_email", "meta__has_microdata_article",
}


def prepare_for_catboost(flat_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    """
    Готовит данные для CatBoost: список плоских dict и имена колонок.
    Булевые значения приводятся к 0/1, как при обучении (из CSV).
    """
    if not flat_rows:
        return [], [_flat_key_to_feature_name(p) for p in FEATURE_SPEC]
    feature_names = [_flat_key_to_feature_name(p) for p in FEATURE_SPEC]
    # Приводим bool к int 0/1, чтобы совпасть с форматом при обучении (CSV)
    normalized: list[dict[str, Any]] = []
    for row in flat_rows:
        out: dict[str, Any] = {}
        for k, v in row.items():
            if k in _BOOL_COLUMNS and isinstance(v, bool):
                out[k] = 1 if v else 0
            else:
                out[k] = v
        normalized.append(out)
    return normalized, feature_names

