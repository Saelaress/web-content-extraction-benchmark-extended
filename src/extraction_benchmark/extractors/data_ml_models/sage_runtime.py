# -*- coding: utf-8 -*-
"""
Рантайм SAGE/Bidirectional для бенчмарка. NodeEncoder / DOMSAGEClassifier должны совпадать с
`6 обучение модели/gnn_model.py`, иначе загрузка `sage_gnn.pt` несовместима.
"""
from __future__ import annotations

import json
import logging
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

_logger = logging.getLogger("wceb.sage")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

try:
    from lxml import html as lxml_html
except ImportError:
    lxml_html = None


LABEL_MAIN = "sure-main-content"
LABEL_TEMPLATE = "sure-template-content"
CLASS_MAIN = 1
CLASS_TEMPLATE = 0
CATEGORICAL_COLUMNS = [
    "node__tag_name",
    "node__parent_tag",
    "node__grandparent_tag",
    "language__language_code",
]
FEATURE_SPEC = [
    "node.dom_depth", "node.dom_depth_norm", "node.tag_name", "node.tag_is_a", "node.tag_is_div",
    "node.tag_is_p", "node.tag_is_heading", "node.tag_is_article", "node.tag_is_nav", "node.tag_is_footer",
    "node.tag_is_header", "node.parent_tag", "node.grandparent_tag", "node.has_parent_article",
    "node.is_excluded_tag", "node.num_children",
    "subtree.tag_count", "subtree.num_leaves", "subtree.text_length_chars", "subtree.word_count",
    "subtree.link_count", "subtree.link_text_length", "subtree.link_text_ratio",
    "subtree.text_without_links_ratio", "subtree.words_per_tag", "subtree.words_per_leaf",
    "subtree.chars_per_descendant", "subtree.links_per_descendant", "subtree.children_ratio",
    "text.has_visible_text", "text.is_whitespace_only", "text.has_only_links", "text.digit_ratio",
    "text.r_punctuation", "text.ends_with_punctuation", "text.num_lines", "text.avg_word_length",
    "text.avg_sentence_length", "text.nlp_comma_density",
    "meta.has_email", "meta.has_microdata_article", "meta.image_caption_ratio", "meta.list_internal_link_ratio",
    "language.language_code", "language.language_confidence",
    "def31.word_ratio", "def31.hyperlink_ratio", "def31.children_ratio_binary", "def31.position_ratio",
]
UNKNOWN_CATEGORY = "__unknown__"
_LEAKAGE_CLASSES = frozenset({LABEL_MAIN, LABEL_TEMPLATE})
TAG_UNK_INDEX = 0


class NodeEncoder(nn.Module):
    """
    Кодировщик узла DOM:
      1. e_tag  = Embedding(tag_index)                          [embed_dim]
      2. e_class = proj_class(FastText avg class words)         [embed_dim]
      3. h_tag_class = ReLU(tag_class_merge([e_tag, e_class]))  [embed_dim]
      4. h_text = ReLU(proj_text(FastText visible text))        [embed_dim]
      5. h_textual = ReLU(merge([h_tag_class, h_text]))         [hidden_dim]
      6. h_num  = ReLU(proj_num(числовые признаки))             [hidden_dim]
      7. out = h_textual + h_num
    """

    def __init__(
        self,
        ft_dim: int,
        embed_dim: int,
        num_tag_embeddings: int,
        num_numeric: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.tag_embed = nn.Embedding(num_tag_embeddings, embed_dim)
        self.proj_class = nn.Linear(ft_dim, embed_dim)
        self.tag_class_merge = nn.Linear(2 * embed_dim, embed_dim)
        self.proj_text = nn.Linear(ft_dim, embed_dim)
        self.merge = nn.Linear(2 * embed_dim, hidden_dim)
        self.proj_num = nn.Linear(num_numeric, hidden_dim)

    def forward(
        self,
        x_tag: torch.Tensor,
        x_text: torch.Tensor,
        x_class: torch.Tensor,
        x_num: torch.Tensor,
    ) -> torch.Tensor:
        e_tag = self.tag_embed(x_tag.long())
        e_class = self.proj_class(x_class)
        h_tag_class = F.relu(self.tag_class_merge(torch.cat([e_tag, e_class], dim=-1)))
        h_text = F.relu(self.proj_text(x_text))
        h_textual = F.relu(self.merge(torch.cat([h_tag_class, h_text], dim=-1)))
        h_num = F.relu(self.proj_num(x_num))
        return h_textual + h_num


class DOMSAGEClassifier(nn.Module):
    """child→parent рёбра: родитель агрегирует детей."""

    def __init__(
        self,
        ft_dim: int,
        embed_dim: int,
        num_tag_embeddings: int,
        num_numeric: int,
        hidden_dim: int,
        num_gnn_layers: int,
        num_classes: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = NodeEncoder(
            ft_dim, embed_dim, num_tag_embeddings, num_numeric, hidden_dim
        )
        self.dropout = nn.Dropout(dropout)
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_gnn_layers):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(
        self,
        x_tag: torch.Tensor,
        x_text: torch.Tensor,
        x_class: torch.Tensor,
        x_num: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        h = F.relu(self.encoder(x_tag, x_text, x_class, x_num))
        for conv, bn in zip(self.convs, self.bns):
            h_res = h
            h = conv(h, edge_index)
            h = bn(h)
            h = self.dropout(F.relu(h)) + h_res
        return self.head(h)


class BiDirSAGEClassifier(nn.Module):
    """
    Каждый слой делает два прохода параллельно:
      - восходящий  (child→parent, edge_index)
      - нисходящий  (parent→child, flip(edge_index))
    Результаты конкатенируются и проецируются обратно в hidden_dim.
    """

    def __init__(
        self,
        ft_dim: int,
        embed_dim: int,
        num_tag_embeddings: int,
        num_numeric: int,
        hidden_dim: int,
        num_layers: int,
        num_classes: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = NodeEncoder(ft_dim, embed_dim, num_tag_embeddings, num_numeric, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.up_convs   = nn.ModuleList([SAGEConv(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.down_convs = nn.ModuleList([SAGEConv(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.projs      = nn.ModuleList([nn.Linear(2 * hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.bns        = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(
        self,
        x_tag: torch.Tensor,
        x_text: torch.Tensor,
        x_class: torch.Tensor,
        x_num: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        edge_index_down = edge_index[[1, 0], :] if edge_index.size(1) > 0 else edge_index
        h = F.relu(self.encoder(x_tag, x_text, x_class, x_num))
        for up_conv, down_conv, proj, bn in zip(self.up_convs, self.down_convs, self.projs, self.bns):
            h_res = h
            h_up   = up_conv(h, edge_index)
            h_down = down_conv(h, edge_index_down)
            h_merged = proj(torch.cat([h_up, h_down], dim=-1))
            h = self.dropout(F.relu(bn(h_merged))) + h_res
        return self.head(h)


def _repo_root() -> Path:
    """Корень репозитория web-content-extraction-benchmark (каталог с `src/`)."""
    return Path(__file__).resolve().parents[4]


def _apply_dom_annotation(html: str) -> str:
    """Проставляет data-ml на узлах перед инференсом (локальный annotate_dom в этом пакете)."""
    if os.environ.get("WCEB_SKIP_DOM_ANNOTATE", "").strip().lower() in ("1", "true", "yes"):
        return html
    try:
        from .annotate_dom import annotate_html

        return annotate_html(html)
    except Exception as e:
        _logger.debug("annotate_html: %s", e)
        return html


def _models_dir() -> Path:
    env = os.environ.get("WCEB_SAGE_MODELS_DIR", "").strip()
    if env:
        return Path(env).expanduser()
    return _repo_root() / "third-party" / "sage-models"


def _get_nested(data: dict[str, Any], path: str) -> Any:
    parts = path.split(".", 1)
    if len(parts) == 1:
        return data.get(path)
    head, tail = parts
    sub = data.get(head)
    if sub is None or not isinstance(sub, dict):
        return None
    return _get_nested(sub, tail)


def _flat_name(path: str) -> str:
    return path.replace(".", "__")


def _data_ml_to_flat_row(data_ml: dict[str, Any]) -> dict[str, Any]:
    categorical = {"node.tag_name", "node.parent_tag", "node.grandparent_tag", "language.language_code"}
    bool_paths = {
        "node.tag_is_a", "node.tag_is_div", "node.tag_is_p", "node.tag_is_heading", "node.tag_is_article",
        "node.tag_is_nav", "node.tag_is_footer", "node.tag_is_header", "node.has_parent_article", "node.is_excluded_tag",
        "text.has_visible_text", "text.is_whitespace_only", "text.has_only_links", "text.ends_with_punctuation",
        "meta.has_email", "meta.has_microdata_article",
    }
    row: dict[str, Any] = {}
    for path in FEATURE_SPEC:
        v = _get_nested(data_ml, path)
        k = _flat_name(path)
        if v is None:
            row[k] = "" if path in categorical else (False if path in bool_paths else 0)
        else:
            row[k] = v
    for k, v in list(row.items()):
        if isinstance(v, bool):
            row[k] = 1 if v else 0
    return row


def _to_numeric_matrix(flat_rows: list[dict[str, Any]], label_encoders: dict[str, Any]) -> np.ndarray:
    if not flat_rows:
        return np.array([], dtype=np.float64).reshape(0, len(FEATURE_SPEC))
    feat_names = [_flat_name(p) for p in FEATURE_SPEC]
    cols = {n: [] for n in feat_names}
    for r in flat_rows:
        for n in feat_names:
            cols[n].append(r.get(n))
    out = []
    for n in feat_names:
        c = cols[n]
        if n in CATEGORICAL_COLUMNS:
            le = label_encoders[n]
            mapped = []
            classes = set(le.classes_.tolist()) if hasattr(le, "classes_") else set()
            for x in c:
                s = str(x) if x != "" else ""
                mapped.append(s if s in classes else UNKNOWN_CATEGORY)
            out.append(le.transform(mapped).astype(np.float64))
        else:
            out.append(np.array(c, dtype=np.float64))
    return np.column_stack(out).astype(np.float64)


def _tag_name(el) -> str:
    if el is None or getattr(el, "tag", None) is None or callable(getattr(el, "tag", None)) or not isinstance(el.tag, str):
        return ""
    return str(el.tag).lower()


def _extract_data_ml(el) -> dict[str, Any] | None:
    raw = el.get("data-ml")
    if not raw:
        return None
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None


def _iter_nodes(root):
    for el in root.iter():
        if _tag_name(el) == "":
            continue
        d = _extract_data_ml(el)
        if d is None:
            continue
        yield el, d


def _build_edge_index(elements: list[Any]) -> torch.Tensor:
    id2i = {id(e): i for i, e in enumerate(elements)}
    src: list[int] = []
    dst: list[int] = []
    for i, el in enumerate(elements):
        p = el.getparent()
        while p is not None:
            j = id2i.get(id(p))
            if j is not None:
                src.append(i)
                dst.append(j)
                break
            p = p.getparent()
    if not src:
        return torch.zeros((2, 0), dtype=torch.long)
    return torch.tensor([src, dst], dtype=torch.long)


def _sanitize_classes(class_attr: str | list[str] | None) -> list[str]:
    if not class_attr:
        return []
    if isinstance(class_attr, list):
        raw_tokens = class_attr
    else:
        raw_tokens = str(class_attr).split()
    tokens = [t.strip() for t in raw_tokens if t and str(t).strip()]
    return [t for t in tokens if t not in _LEAKAGE_CLASSES]


def _collect_text(el, parts: list[str]) -> None:
    tag = _tag_name(el)
    if tag in ("script", "style", "noscript", "iframe", "svg"):
        return
    if el.text:
        parts.append(el.text)
    for c in el:
        _collect_text(c, parts)
        if c.tail:
            parts.append(c.tail)


def _visible_text(el) -> str:
    parts: list[str] = []
    _collect_text(el, parts)
    return " ".join(parts).strip()


def _tokenize_words(text: str) -> list[str]:
    if not text or not text.strip():
        return []
    return re.findall(r"[^\W\d_]+|\d+", text.lower(), flags=re.UNICODE)


def _avg_word_vectors(ft_model, words: list[str]) -> np.ndarray:
    """Точное воспроизведение логики обучения (sage_features.py)."""
    dim = int(ft_model.get_dimension())
    if not words:
        return np.zeros(dim, dtype=np.float32)
    vecs = []
    for w in words:
        try:
            vecs.append(np.asarray(ft_model.get_word_vector(w), dtype=np.float32))
        except Exception:
            continue
    if not vecs:
        return np.zeros(dim, dtype=np.float32)
    return np.mean(np.stack(vecs, axis=0), axis=0).astype(np.float32)


@lru_cache(maxsize=1)
def _load_fasttext_and_prep() -> tuple[Any, dict]:
    """Загружает FastText и препроцессоры один раз — используется обеими моделями."""
    import fasttext

    md = _models_dir()
    prep = torch.load(md / "sage_preprocessors.pt", map_location="cpu", weights_only=False)
    ft_path = os.environ.get("FASTTEXT_MODEL_PATH", "").strip() or str(prep.get("fasttext_path") or "").strip()
    default_ft = md / "cc.en.300.bin"
    cand = Path(ft_path).expanduser() if ft_path else default_ft
    if not ft_path or not cand.is_file():
        cand = default_ft
    if not cand.is_file():
        raise FileNotFoundError(
            "Не найден файл FastText (.bin). Укажите FASTTEXT_MODEL_PATH или положите cc.en.300.bin в "
            f"{default_ft.parent}"
        )
    ft_model = fasttext.load_model(str(cand.resolve()))
    num_tag_embeddings = int(prep.get("num_tag_embeddings", 0))
    if num_tag_embeddings <= 0 or "tag_stoi" not in prep:
        raise ValueError(
            "В sage_preprocessors.pt нет tag_stoi/num_tag_embeddings. "
            "Пересоберите препроцессоры и скопируйте артефакты в third-party/sage-models/."
        )
    return ft_model, prep


@lru_cache(maxsize=1)
def _load_sage_bundle():
    md = _models_dir()
    ft_model, prep = _load_fasttext_and_prep()
    ckpt = torch.load(md / "sage_gnn.pt", map_location="cpu", weights_only=False)
    model = DOMSAGEClassifier(
        ft_dim=int(prep["ft_dim"]),
        embed_dim=int(prep.get("embed_dim", 16)),
        num_tag_embeddings=int(prep["num_tag_embeddings"]),
        num_numeric=int(prep["num_numeric"]),
        hidden_dim=int(prep.get("hidden_dim", 64)),
        num_gnn_layers=int(prep.get("num_gnn_layers", 2)),
        num_classes=2,
        dropout=0.1,
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ft_model, prep


@lru_cache(maxsize=1)
def _load_bidirectional_bundle():
    md = _models_dir()
    ft_model, prep = _load_fasttext_and_prep()
    ckpt = torch.load(md / "bidirectional_gnn.pt", map_location="cpu", weights_only=False)
    model = BiDirSAGEClassifier(
        ft_dim=int(prep["ft_dim"]),
        embed_dim=int(ckpt.get("embed_dim", 32)),
        num_tag_embeddings=int(prep["num_tag_embeddings"]),
        num_numeric=int(prep["num_numeric"]),
        hidden_dim=int(ckpt.get("hidden_dim", 128)),
        num_layers=int(ckpt.get("num_layers", 3)),
        num_classes=2,
        dropout=0.1,
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ft_model, prep


def _compute_node_features(
    elements: list[Any],
    data_mls: list[dict[str, Any]],
    ft_model: Any,
    label_encoders: dict,
    scaler: Any,
    tag_stoi: dict[str, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Вычисляет (x_tag, x_text, x_class, x_num) для списка DOM-элементов.
    FastText-векторы: один вызов get_sentence_vector на узел вместо цикла по словам.
    """
    tag_idx: list[int] = []
    text_vecs: list[np.ndarray] = []
    class_vecs: list[np.ndarray] = []

    for el in elements:
        tw = _tag_name(el)
        tag_idx.append(int(tag_stoi.get(tw, TAG_UNK_INDEX)) if tw else TAG_UNK_INDEX)
        text_vecs.append(_avg_word_vectors(ft_model, _tokenize_words(_visible_text(el))))
        class_vecs.append(_avg_word_vectors(ft_model, _sanitize_classes(el.get("class"))))

    flat_rows = [_data_ml_to_flat_row(d) for d in data_mls]
    x_num = _to_numeric_matrix(flat_rows, label_encoders=label_encoders)
    x_num = scaler.transform(x_num.astype(np.float64))

    x_tag = torch.tensor(tag_idx, dtype=torch.long)
    x_text = torch.from_numpy(np.stack(text_vecs, axis=0)).float()
    x_class = torch.from_numpy(np.stack(class_vecs, axis=0)).float()
    x_num_t = torch.from_numpy(x_num.astype(np.float32))
    return x_tag, x_text, x_class, x_num_t


def _predict_labels_sage(root) -> tuple[list[Any], list[int]]:
    pairs = list(_iter_nodes(root)) if root is not None else []
    if not pairs:
        return [], []
    elements, data_mls = zip(*pairs)
    elements = list(elements)
    data_mls = list(data_mls)
    model, ft_model, prep = _load_sage_bundle()
    edge_index = _build_edge_index(elements)
    x_tag, x_text, x_class, x_num_t = _compute_node_features(
        elements, data_mls, ft_model, prep["label_encoders"], prep["scaler"], prep["tag_stoi"]
    )
    with torch.no_grad():
        pred = model(x_tag, x_text, x_class, x_num_t, edge_index).argmax(dim=-1).cpu().numpy().astype(int).tolist()
    return elements, pred


def _predict_labels_bidirectional(root) -> tuple[list[Any], list[int]]:
    pairs = list(_iter_nodes(root)) if root is not None else []
    if not pairs:
        return [], []
    elements, data_mls = zip(*pairs)
    elements = list(elements)
    data_mls = list(data_mls)
    model, ft_model, prep = _load_bidirectional_bundle()
    edge_index = _build_edge_index(elements)
    x_tag, x_text, x_class, x_num_t = _compute_node_features(
        elements, data_mls, ft_model, prep["label_encoders"], prep["scaler"], prep["tag_stoi"]
    )
    with torch.no_grad():
        pred = model(x_tag, x_text, x_class, x_num_t, edge_index).argmax(dim=-1).cpu().numpy().astype(int).tolist()
    return elements, pred


def _extract_text_from_labels(elements: list[Any], labels: list[int]) -> str:
    if not elements or not labels:
        return ""
    main_els = [el for el, lab in zip(elements, labels) if lab == CLASS_MAIN]
    if not main_els:
        return ""
    # O(M×depth) вместо O(M²): innermost = main-узлы без main-потомков.
    # Для каждого main-узла идём вверх по предкам — если предок тоже main,
    # помечаем предка как «имеющего main-потомка» (т.е. он не innermost).
    main_ids = {id(el) for el in main_els}
    has_main_descendant: set[int] = set()
    for el in main_els:
        p = el.getparent()
        while p is not None:
            pid = id(p)
            if pid in main_ids:
                has_main_descendant.add(pid)
                break
            p = p.getparent()
    innermost = [el for el in main_els if id(el) not in has_main_descendant]
    parts = []
    for el in innermost:
        t = _visible_text(el)
        if t:
            parts.append(t)
    text = "\n".join(parts).strip()
    text = re.sub(r" +", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def extract_with_sage(html: str, page_id: str = "") -> str:
    if lxml_html is None:
        _logger.warning("[sage] lxml не установлен")
        return ""
    raw = html or ""
    if not raw.strip():
        return ""
    annotated = _apply_dom_annotation(raw)
    try:
        doc = lxml_html.fromstring(annotated)
    except Exception:
        try:
            doc = lxml_html.fromstring(annotated.encode("utf-8", errors="replace"))
        except Exception:
            _logger.warning("[sage] (%s) lxml не смог распарсить HTML", page_id)
            return ""
    root = doc.getroottree().getroot()
    if root is None:
        return ""
    try:
        elements, pred = _predict_labels_sage(root)
    except Exception as e:
        _logger.warning("[sage] (%s) ошибка предсказания: %s", page_id, e)
        return ""
    _logger.debug("[sage] (%s) nodes=%d predicted_main=%d", page_id, len(elements), sum(p == CLASS_MAIN for p in pred))
    return _extract_text_from_labels(elements, pred)


def extract_with_bidirectional(html: str, page_id: str = "") -> str:
    if lxml_html is None:
        _logger.warning("[bidirectional] lxml не установлен")
        return ""
    raw = html or ""
    if not raw.strip():
        return ""
    annotated = _apply_dom_annotation(raw)
    try:
        doc = lxml_html.fromstring(annotated)
    except Exception:
        try:
            doc = lxml_html.fromstring(annotated.encode("utf-8", errors="replace"))
        except Exception:
            _logger.warning("[bidirectional] (%s) lxml не смог распарсить HTML", page_id)
            return ""
    root = doc.getroottree().getroot()
    if root is None:
        return ""
    try:
        elements, pred = _predict_labels_bidirectional(root)
    except Exception as e:
        _logger.warning("[bidirectional] (%s) ошибка предсказания: %s", page_id, e)
        return ""
    return _extract_text_from_labels(elements, pred)
