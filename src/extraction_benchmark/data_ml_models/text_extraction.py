# -*- coding: utf-8 -*-
"""
Извлечение текста из HTML по предсказаниям моделей поверх data-ml.

Основано на дипломном модуле text_extraction.py, адаптировано для
встроенного использования в web-content-extraction-benchmark.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Iterator

try:
    from lxml import html as lxml_html
    from lxml.etree import _Element
except ImportError:  # pragma: no cover
    lxml_html = None
    _Element = None


def _get_tag_name(el: _Element) -> str:
    if el is None or el.tag is None or callable(el.tag) or not isinstance(el.tag, str):
        return ""
    return (el.tag if isinstance(el.tag, str) else str(el.tag)).lower()


def _is_inside_technical_subtree(el: _Element) -> bool:
    """True, если el — script/style/noscript/iframe/svg или внутри такого узла."""
    current = el
    while current is not None:
        tag = _get_tag_name(current)
        if tag in ("script", "style", "noscript", "iframe", "svg"):
            return True
        current = current.getparent()
    return False


def _iter_text_segments_with_owner(root: _Element) -> Iterator[tuple[str, _Element]]:
    """
    Обходит дерево и выдаёт (текст, элемент-владелец).
    el.text принадлежит el; child.tail принадлежит el (родителю).
    Пропускает текст из script/style/noscript/iframe/svg.
    """
    for el in root.iter():
        if _is_inside_technical_subtree(el):
            continue
        if el.text:
            t = el.text.strip()
            if t:
                yield t, el
        for child in el:
            if child.tail:
                t = child.tail.strip()
                if t:
                    yield t, el


def get_text_leaves_from_html(html_path: Path, encoding: str = "utf-8") -> list[tuple[str, int]]:
    """
    Извлекает текстовые листья и индекс узла (data-ml), которому они принадлежат.

    Порядок узлов должен совпадать с порядком обхода узлов с data-ml
    в модуле инференса.
    """
    if lxml_html is None:
        raise RuntimeError("Требуется lxml. Установите: pip install lxml")

    path_str = str(html_path.resolve())
    if sys.platform == "win32" and len(path_str) >= 259 and not path_str.startswith("\\\\?\\"):
        path_str = "\\\\?\\" + path_str

    with open(path_str, "rb") as f:
        raw = f.read()
    content = raw.decode(encoding, errors="replace")

    doc = lxml_html.fromstring(content)
    root = doc.getroottree().getroot()
    if root is None:
        return []

    # Здесь мы ожидаем, что внешний код даст сопоставление
    # "элемент с data-ml" -> индекс узла; поэтому просто нумеруем встреченные элементы.
    nodes_with_data_ml: list[_Element] = []
    for el in root.iter():
        if _get_tag_name(el) == "":
            continue
        if el.get("data-ml"):
            nodes_with_data_ml.append(el)
    element_to_index = {id(el): i for i, el in enumerate(nodes_with_data_ml)}

    result: list[tuple[str, int]] = []
    for text, owner_el in _iter_text_segments_with_owner(root):
        current = owner_el
        idx = None
        while current is not None:
            eid = id(current)
            if eid in element_to_index:
                idx = element_to_index[eid]
                break
            current = current.getparent()
        if idx is not None:
            result.append((text, idx))
    return result


def combine_predicted_text(leaves: list[tuple[str, int]], predictions: list[int]) -> str:
    """
    Объединяет текст листьев, у которых prediction == 1 (main content).

    leaves: список (текст, индекс_узла)
    predictions: список предсказаний по индексу узла (0=template, 1=main)
    """
    parts: list[str] = []
    for text, node_idx in leaves:
        if node_idx < len(predictions) and predictions[node_idx] == 1:
            parts.append(text)
    return "\n".join(parts) if parts else ""


def extract_text_from_predictions(
    html_path: Path,
    predictions: list[int],
    encoding: str = "utf-8",
) -> str:
    """
    Извлекает текст из HTML по предсказаниям модели (1=main).
    
    Порядок узлов и предсказаний должен соответствовать порядку обхода
    узлов с data-ml в документе.
    """
    leaves = get_text_leaves_from_html(html_path, encoding=encoding)
    return combine_predicted_text(leaves, predictions)


def _is_descendant(el: _Element, anc: _Element) -> bool:
    """True, если el — потомок anc (или совпадает с anc)."""
    current = el
    while current is not None:
        if current is anc:
            return True
        current = current.getparent()
    return False


def _collect_text_recursive(el: _Element, parts: list[str]) -> None:
    """Рекурсивно собирает текст, пропуская Comment/PI и технические поддеревья."""
    tag = _get_tag_name(el)
    if tag == "":
        return
    if tag in ("script", "style", "noscript", "iframe", "svg"):
        return
    if el.text:
        parts.append(el.text)
    for child in el:
        _collect_text_recursive(child, parts)
        if child.tail:
            parts.append(child.tail)


def _get_text_content_clean(el: _Element) -> str:
    """Извлекает текст узла, исключая содержимое script/style/noscript/iframe/svg и комментарии."""
    parts: list[str] = []
    _collect_text_recursive(el, parts)
    return " ".join(parts).strip()


def extract_text_from_labeled_nodes_full(
    html_path: Path,
    labels: list[int],
    encoding: str = "utf-8",
    block_sep: str = "\n",
) -> str:
    """
    Извлекает текст только из узлов с меткой 1 (main content).

    Для каждого такого узла берётся полный текст узла (text_content), без дублирования:
    если у узла есть потомок с data-ml и меткой 1, текст потомка не добавляется отдельно.
    """
    if lxml_html is None:
        raise RuntimeError("Требуется lxml. Установите: pip install lxml")

    path_str = str(html_path.resolve())
    if sys.platform == "win32" and len(path_str) >= 259 and not path_str.startswith("\\\\?\\"):
        path_str = "\\\\?\\" + path_str

    with open(path_str, "rb") as f:
        raw = f.read()
    content = raw.decode(encoding, errors="replace")

    doc = lxml_html.fromstring(content)
    root = doc.getroottree().getroot()
    if root is None:
        return ""

    nodes_list: list[_Element] = []
    for el in root.iter():
        if _get_tag_name(el) == "":
            continue
        if el.get("data-ml"):
            nodes_list.append(el)

    if len(nodes_list) != len(labels):
        print(
            f"WARN: несовпадение числа узлов и меток ({len(nodes_list)} vs {len(labels)}) для {html_path}",
            file=sys.stderr,
        )
        return ""

    main_els = [el for el, lab in zip(nodes_list, labels) if lab == 1]

    innermost: list[_Element] = []
    for el in main_els:
        if not any(_is_descendant(other, el) and other is not el for other in main_els):
            innermost.append(el)

    parts: list[str] = []
    for el in innermost:
        t = _get_text_content_clean(el)
        if t:
            parts.append(t)
    text = block_sep.join(parts) if parts else ""
    text = re.sub(r" +", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()

