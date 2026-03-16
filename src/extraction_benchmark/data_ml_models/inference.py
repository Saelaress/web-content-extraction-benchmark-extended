# -*- coding: utf-8 -*-
"""
Инференс-обёртка для интеграции моделей (LogReg / RF / CatBoost)
в бенчмарк web-content-extraction:

HTML (с атрибутами data-ml на узлах) -> извлечённый plaintext.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from extraction_benchmark.paths import THIRD_PARTY_PATH
from .text_extraction import extract_text_from_labeled_nodes_full
from .vectorize import (
    data_ml_to_flat_row,
    prepare_for_catboost,
    prepare_for_sklearn,
)

try:
    from lxml import html as lxml_html
    from lxml.etree import _Element
except ImportError:  # pragma: no cover
    lxml_html = None
    _Element = None


MODELS_ROOT = Path(THIRD_PARTY_PATH) / "data-ml-models"


def _get_tag_name(el: _Element) -> str:
    if el is None or el.tag is None or callable(el.tag) or not isinstance(el.tag, str):
        return ""
    return (el.tag if isinstance(el.tag, str) else str(el.tag)).lower()


def _extract_data_ml(el: _Element) -> dict[str, Any] | None:
    """Извлечь и распарсить JSON из атрибута data-ml."""
    raw = el.get("data-ml")
    if not raw:
        return None
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None


def _iter_nodes_with_data_ml(root: _Element):
    """Итерирует узлы с data-ml в порядке обхода документа, возвращает (data_ml_dict, element)."""
    for el in root.iter():
        if _get_tag_name(el) == "":
            continue
        data_ml = _extract_data_ml(el)
        if data_ml is None:
            continue
        yield data_ml, el


def _html_to_data_ml_rows(html: str) -> tuple[list[dict[str, Any]], list[_Element]]:
    """
    Парсит HTML-строку и возвращает:
    - список data-ml dictов (в порядке обхода),
    - список соответствующих элементов.
    """
    if lxml_html is None:
        raise RuntimeError("Требуется lxml. Установите: pip install lxml")

    doc = lxml_html.fromstring(html)
    root = doc.getroottree().getroot()
    if root is None:
        return [], []

    rows: list[dict[str, Any]] = []
    elements: list[_Element] = []
    for data_ml, el in _iter_nodes_with_data_ml(root):
        rows.append(data_ml)
        elements.append(el)
    return rows, elements


@lru_cache(maxsize=1)
def _load_logreg():
    import joblib

    scaler = joblib.load(MODELS_ROOT / "scaler.joblib")
    model = joblib.load(MODELS_ROOT / "logreg.joblib")
    return scaler, model


@lru_cache(maxsize=1)
def _load_rf():
    import joblib

    model = joblib.load(MODELS_ROOT / "rf.joblib")
    return model


@lru_cache(maxsize=1)
def _load_catboost():
    from catboost import CatBoostClassifier

    model = CatBoostClassifier()
    model.load_model(str(MODELS_ROOT / "catboost.cbm"))
    return model


def _predict_labels_logreg(flat_rows: list[dict[str, Any]]) -> list[int]:
    if not flat_rows:
        return []
    scaler, model = _load_logreg()
    X, _, _ = prepare_for_sklearn(flat_rows)
    X_scaled = scaler.transform(X.astype(np.float64))
    y_pred = model.predict(X_scaled)
    return y_pred.astype(int).tolist()


def _predict_labels_rf(flat_rows: list[dict[str, Any]]) -> list[int]:
    if not flat_rows:
        return []
    model = _load_rf()
    X, _, _ = prepare_for_sklearn(flat_rows)
    y_pred = model.predict(X.astype(np.float64))
    return y_pred.astype(int).tolist()


def _predict_labels_catboost(flat_rows: list[dict[str, Any]]) -> list[int]:
    if not flat_rows:
        return []
    import pandas as pd

    model = _load_catboost()
    rows_for_cb, feature_names = prepare_for_catboost(flat_rows)
    df = pd.DataFrame(rows_for_cb, columns=feature_names)
    y_pred = model.predict(df)
    if hasattr(y_pred, "ravel"):
        y_pred = y_pred.ravel()
    return np.array(y_pred, dtype=int).tolist()


def _build_labels_for_elements(
    elements: list[_Element], node_labels: list[int]
) -> list[int]:
    """
    Преобразует список меток по узлам data-ml в список меток той же длины
    (один к одному: node i -> label[i]).
    """
    if not elements or not node_labels:
        return []
    if len(elements) != len(node_labels):
        n = min(len(elements), len(node_labels))
        node_labels = node_labels[:n]
        elements = elements[:n]
    return node_labels


def extract_with_model(html: str, page_id: str, model_name: str) -> str:
    """
    Главная точка входа: HTML + page_id -> plaintext.

    model_name: 'rf' | 'lr' | 'catboost'
    """
    del page_id

    rows_data, elements = _html_to_data_ml_rows(html)
    if not rows_data or not elements:
        return ""

    flat_rows = [data_ml_to_flat_row(d) for d in rows_data]

    if model_name == "lr":
        node_labels = _predict_labels_logreg(flat_rows)
    elif model_name == "rf":
        node_labels = _predict_labels_rf(flat_rows)
    elif model_name == "catboost":
        node_labels = _predict_labels_catboost(flat_rows)
    else:
        raise ValueError(f"Неизвестная модель: {model_name}")

    labels_for_nodes = _build_labels_for_elements(elements, node_labels)

    # Собираем текст по меткам узлов (как для ground truth).
    from tempfile import NamedTemporaryFile

    with NamedTemporaryFile(suffix=".html", delete=True) as tmp:
        tmp.write(html.encode("utf-8", errors="replace"))
        tmp.flush()
        tmp_path = Path(tmp.name)
        text = extract_text_from_labeled_nodes_full(
            tmp_path, labels_for_nodes, encoding="utf-8", block_sep="\n"
        )
    return text or ""

