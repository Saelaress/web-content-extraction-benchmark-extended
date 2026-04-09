# -*- coding: utf-8 -*-
"""
Аннотация HTML-узлов атрибутом data-ml
(признаки из одной страницы, без class/id) через bs4 + html5lib.
Используется экстрактором PyG перед инференсом на «голом» HTML бенчмарка.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

try:
    from bs4 import BeautifulSoup
    from bs4.element import Tag
except ImportError:
    BeautifulSoup = None
    Tag = None

try:
    from langdetect import detect_langs, LangDetectException
    _HAS_LANGDETECT = True
except ImportError:
    _HAS_LANGDETECT = False
    LangDetectException = Exception  # noqa: A001


SCHEMA_VERSION = 1

EXCLUDED_TAGS = frozenset(
    {"script", "style", "noscript", "iframe", "svg", "path", "object", "embed"}
)

RE_EMAIL = re.compile(r"\S+@\S+\.\S+")
PUNCTUATION_CHARS = set(".,!?;:\u2014\u2013-")


def _tokenize_words(text: str) -> list[str]:
    if not text or not text.strip():
        return []
    return [s for s in re.split(r"\s+", text.strip()) if s]


def _sentence_count(text: str) -> int:
    if not text or not text.strip():
        return 0
    parts = re.split(r"[.!?]+", text.strip())
    return max(1, len([p for p in parts if p.strip()]))


def _iter_tags(element: Any):
    if isinstance(element, Tag):
        yield element
    for el in getattr(element, "descendants", []):
        if isinstance(el, Tag):
            yield el


def _child_tags(element: Tag) -> list[Tag]:
    return [c for c in element.children if isinstance(c, Tag)]


def _get_depth(element: Tag, root: Any) -> int:
    depth = 0
    cur = element
    while cur is not None and cur != root:
        depth += 1
        cur = cur.parent
    return depth


def _get_distance(from_el: Tag, to_el: Tag) -> int:
    d = 0
    cur = from_el
    while cur is not None and cur != to_el:
        d += 1
        cur = cur.parent
    return d


def _get_subtree_text(element: Tag) -> str:
    return (element.get_text(" ", strip=True) or "").strip()


def _get_link_text_length(element: Tag) -> int:
    total = 0
    for a in element.find_all("a"):
        total += len((a.get_text(" ", strip=True) or "").strip())
    return total


def _count_leaves(element: Tag) -> int:
    count = 0
    for el in _iter_tags(element):
        has_element_child = any(True for _ in _child_tags(el))
        if not has_element_child:
            count += 1
    return count


def _tag_name(el: Any) -> str:
    if not isinstance(el, Tag):
        return ""
    return (el.name or "").lower()


def _has_ancestor_article(element: Tag, root: Any) -> bool:
    cur = element.parent
    while cur is not None and cur != root:
        if _tag_name(cur) == "article":
            return True
        cur = cur.parent
    return False


def _has_microdata_article(element: Tag) -> bool:
    for el in _iter_tags(element):
        itemprop = (el.get("itemprop") or "").strip()
        if "articlebody" in itemprop.lower() or "article" in itemprop.lower():
            return True
    return False


def _image_caption_ratio(element: Tag) -> float:
    imgs = element.find_all("img")
    figcaps = element.find_all("figcaption")
    n_img = len(imgs)
    n_fig = len(figcaps)
    if n_img == 0:
        return 0.0
    return round(n_fig / n_img, 6)


def _list_internal_link_ratio(element: Tag) -> float:
    lis = element.find_all("li")
    if not lis:
        return 0.0
    with_links = sum(1 for li in lis if li.find("a") is not None)
    return round(with_links / len(lis), 6)


def _word_ratio_def31(element: Tag) -> float:
    total = 0.0
    for el in _iter_tags(element):
        parent = el.parent
        if parent is not None and _tag_name(parent) == "a":
            continue
        has_child_el = any(True for _ in _child_tags(el))
        if has_child_el:
            continue
        text = (el.get_text(" ", strip=True) or "").strip()
        w = len(_tokenize_words(text))
        if w == 0:
            continue
        dist = max(1, _get_distance(el, element))
        total += w / dist
    return round(total, 6)


def _language_features(text: str) -> tuple[str, float]:
    if not _HAS_LANGDETECT or not text:
        return ("", 0.0)
    stripped = text.strip()
    if len(stripped) < 10:
        return ("", 0.0)
    try:
        langs = detect_langs(stripped)
        if not langs:
            return ("", 0.0)
        top = langs[0]
        return (top.lang, round(top.prob, 6))
    except (LangDetectException, Exception):
        return ("", 0.0)


def _compute_subtree_stats(element: Tag) -> dict[str, Any]:
    text = _get_subtree_text(element)
    words = _tokenize_words(text)
    tag_count = sum(1 for _ in _iter_tags(element))
    num_leaves = _count_leaves(element)
    link_count = len(element.find_all("a"))
    link_text_length = _get_link_text_length(element)
    text_length_chars = len(text)
    word_count = len(words)

    return {
        "tag_count": tag_count,
        "num_leaves": max(1, num_leaves),
        "num_leaves_raw": num_leaves,
        "text_length_chars": text_length_chars,
        "word_count": word_count,
        "link_count": link_count,
        "link_text_length": link_text_length,
        "text": text,
        "sentence_count": _sentence_count(text),
    }


def _build_data_ml(
    element: Tag,
    root: Any,
    max_depth: int,
    subtree: dict[str, Any],
) -> dict[str, Any]:
    depth = _get_depth(element, root)
    tag_name = _tag_name(element)
    parent = element.parent if isinstance(element.parent, Tag) else None
    parent_tag = _tag_name(parent) if parent is not None else ""
    grandparent = parent.parent if (parent is not None and isinstance(parent.parent, Tag)) else None
    grandparent_tag = _tag_name(grandparent) if grandparent is not None else ""

    tc = subtree["tag_count"]
    nl = subtree["num_leaves_raw"]
    tl = subtree["text_length_chars"]
    wc = subtree["word_count"]
    lc = subtree["link_count"]
    lt = subtree["link_text_length"]
    text = subtree["text"]
    sent_count = subtree["sentence_count"]

    link_text_ratio = round(lt / max(1, tl), 6)
    text_without_links_ratio = round((tl - lt) / max(1, tl), 6)
    words_per_tag = round(wc / max(1, tc), 6)
    words_per_leaf = round(wc / max(1, nl), 6) if nl else 0.0
    chars_per_descendant = round(tl / max(1, tc), 6)
    links_per_descendant = round(lc / max(1, tc), 6)
    num_children = len(_child_tags(element))
    children_ratio = round(num_children / max(1, tc), 6)

    digit_ratio = round(sum(1 for c in text if c.isdigit()) / max(1, tl), 6)
    punct_count = sum(1 for c in text if c in PUNCTUATION_CHARS)
    r_punctuation = round(punct_count / max(1, tl), 6)
    ends_with_punctuation = bool(text and text[-1] in PUNCTUATION_CHARS)
    num_lines = text.count("\n")
    avg_word_length = round(tl / max(1, wc), 6)
    avg_sentence_length = round(wc / max(1, sent_count), 6) if sent_count else 0.0
    nlp_comma_density = round(text.count(",") / max(1, tl), 6)

    has_visible_text = tl > 0
    is_whitespace_only = not text.strip() if text else True
    has_only_links = (lc > 0 and tl > 0 and lt >= tl * 0.99)

    depth_norm = round(depth / max(1, max_depth), 6)

    # hyperlink_ratio = количество ссылок / количество тегов в поддереве
    # (согласовано с links_per_descendant; узлы без ссылок дают 0.0)
    hyperlink_ratio_def31 = round(lc / max(1, tc), 6)
    children_ratio_binary = 0 if num_children <= 2 else 1
    position_ratio = (
        1.0
        if depth <= max_depth / 2
        else round(max_depth / max(1, depth) - 1.0, 6)
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "node": {
            "dom_depth": depth,
            "dom_depth_norm": depth_norm,
            "tag_name": tag_name,
            "tag_is_a": tag_name == "a",
            "tag_is_div": tag_name == "div",
            "tag_is_p": tag_name == "p",
            "tag_is_heading": tag_name in ("h1", "h2", "h3", "h4", "h5", "h6"),
            "tag_is_article": tag_name == "article",
            "tag_is_nav": tag_name == "nav",
            "tag_is_footer": tag_name == "footer",
            "tag_is_header": tag_name == "header",
            "parent_tag": parent_tag,
            "grandparent_tag": grandparent_tag,
            "has_parent_article": _has_ancestor_article(element, root),
            "is_excluded_tag": tag_name in EXCLUDED_TAGS,
            "num_children": num_children,
        },
        "subtree": {
            "tag_count": tc,
            "num_leaves": subtree["num_leaves"],
            "text_length_chars": tl,
            "word_count": wc,
            "link_count": lc,
            "link_text_length": lt,
            "link_text_ratio": link_text_ratio,
            "text_without_links_ratio": text_without_links_ratio,
            "words_per_tag": words_per_tag,
            "words_per_leaf": words_per_leaf,
            "chars_per_descendant": chars_per_descendant,
            "links_per_descendant": links_per_descendant,
            "children_ratio": children_ratio,
        },
        "text": {
            "has_visible_text": has_visible_text,
            "is_whitespace_only": is_whitespace_only,
            "has_only_links": has_only_links,
            "digit_ratio": digit_ratio,
            "r_punctuation": r_punctuation,
            "ends_with_punctuation": ends_with_punctuation,
            "num_lines": num_lines,
            "avg_word_length": avg_word_length,
            "avg_sentence_length": avg_sentence_length,
            "nlp_comma_density": nlp_comma_density,
        },
        "links": {},
        "meta": {
            "has_email": bool(RE_EMAIL.search(text)),
            "has_microdata_article": _has_microdata_article(element),
            "image_caption_ratio": _image_caption_ratio(element),
            "list_internal_link_ratio": _list_internal_link_ratio(element),
        },
        "language": dict(zip(("language_code", "language_confidence"), _language_features(text))),
        "def31": {
            "word_ratio": _word_ratio_def31(element),
            "hyperlink_ratio": hyperlink_ratio_def31,
            "children_ratio_binary": children_ratio_binary,
            "position_ratio": position_ratio,
        },
    }


def _get_max_depth(root: Any) -> int:
    max_d = 0
    for el in _iter_tags(root):
        d = _get_depth(el, root)
        if d > max_d:
            max_d = d
    return max_d


def annotate_html(html_input: str) -> str:
    if not BeautifulSoup:
        raise RuntimeError(
            "Требуется bs4 + html5lib. Установите: pip install beautifulsoup4 html5lib"
        )

    soup = BeautifulSoup(html_input, "html5lib")
    root = soup
    max_depth = _get_max_depth(root)

    for element in _iter_tags(root):
        subtree = _compute_subtree_stats(element)
        data_ml = _build_data_ml(element, root, max_depth, subtree)
        json_str = json.dumps(data_ml, ensure_ascii=False)
        element["data-ml"] = json_str

    return str(soup)


def annotate_file(path: Path | str, encoding: str = "utf-8") -> str:
    path = Path(path)
    raw = path.read_bytes()
    try:
        html_input = raw.decode(encoding)
    except UnicodeDecodeError:
        html_input = raw.decode("utf-8", errors="replace")
    return annotate_html(html_input)


def main() -> None:
    if len(sys.argv) < 2:
        print(
            "Usage: python -m extraction_benchmark.extractors.data_ml_models.annotate_dom <in.html> [out.html]",
            file=sys.stderr,
        )
        sys.exit(1)
    inp = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    result = annotate_file(inp)
    if out:
        out.write_text(result, encoding="utf-8")
        print(f"Written: {out}")
    else:
        print(result)


if __name__ == "__main__":
    main()
