#!/usr/bin/env python
# Copyright 2026
#
import argparse
import json
import os
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
from bs4 import BeautifulSoup
import nltk
import importlib
import tensorflow_core as tf_core

_orig_np_prod = np.prod


def _safe_np_prod(shape, **kwargs):
    try:
        return _orig_np_prod(shape, **kwargs)
    except (NotImplementedError, TypeError):
        return 0


np.prod = _safe_np_prod

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
sys.path.insert(0, SRC_ROOT)

tf_python = tf_core.python
tf_platform = tf_python.platform
sys.modules.setdefault("tensorflow", tf_core)
sys.modules.setdefault("tensorflow.python", tf_python)
sys.modules.setdefault("tensorflow.python.platform", tf_platform)

from extraction_benchmark.extractors.boilernet.net.preprocess import get_feature_vector, get_leaves, process

BOILERNET_ROOT = os.path.join(SRC_ROOT, "extraction_benchmark", "extractors", "boilernet")
MODEL_PATH = os.path.join(BOILERNET_ROOT, "model.h5")
WORDS_PATH = os.path.join(BOILERNET_ROOT, "words.json")
TAGS_PATH = os.path.join(BOILERNET_ROOT, "tags.json")

_model = None
_word_map = None
_tag_map = None

tf = tf_core

array_ops = importlib.import_module("tensorflow.python.ops.array_ops")
from tensorflow_core.python.framework import tensor_util

_orig_make_tensor_proto = tensor_util.make_tensor_proto


def _patched_make_tensor_proto(values, dtype=None, shape=None, verify_shape=False):
    try:
        return _orig_make_tensor_proto(values, dtype=dtype, shape=shape, verify_shape=verify_shape)
    except NotImplementedError:
        return _orig_make_tensor_proto(values, dtype=dtype, shape=None, verify_shape=verify_shape)


tensor_util.make_tensor_proto = _patched_make_tensor_proto

_orig_constant_if_small = array_ops._constant_if_small


def _patched_constant_if_small(zero, shape, dtype, name):
    try:
        return _orig_constant_if_small(zero, shape, dtype, name)
    except (NotImplementedError, TypeError):
        return array_ops.constant(zero, dtype=dtype, shape=shape, name=name)


array_ops._constant_if_small = _patched_constant_if_small


def _load_model():
    global _model, _word_map, _tag_map
    if _model is not None:
        return _model, _word_map, _tag_map
    _model = tf.keras.models.load_model(MODEL_PATH)
    with open(WORDS_PATH, "r", encoding="utf-8") as f:
        _word_map = json.load(f)
    with open(TAGS_PATH, "r", encoding="utf-8") as f:
        _tag_map = json.load(f)
    nltk.download("punkt", quiet=True)
    return _model, _word_map, _tag_map


def boilernet_extract(html: str) -> str:
    model, word_map, tag_map = _load_model()

    tags = defaultdict(int)
    words = defaultdict(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        doc = BeautifulSoup(html, features="html5lib")

    processed = process(doc, tags, words)
    if not processed:
        return ""

    inputs = [get_feature_vector(w, t, word_map, tag_map) for w, t, _ in processed]
    inputs = np.expand_dims(np.stack(inputs), 0)
    predicted = np.around(model.predict(inputs, verbose=0))

    main_content = []
    doc = BeautifulSoup(html, features="html5lib")
    root = doc.find_all("html")
    if root:
        root = root[0]
    else:
        root = doc
    for i, (leaf, _, _) in enumerate(get_leaves(root)):
        if predicted[0, i, 0]:
            main_content.append(leaf)

    return "\n".join(main_content).strip()


def main():
    parser = argparse.ArgumentParser(description="Run Boilernet inference in a TF1 venv")
    parser.add_argument("html_path", type=str, help="Path to HTML file that should be processed")
    args = parser.parse_args()

    html = Path(args.html_path).read_text(encoding="utf-8", errors="replace")
    text = boilernet_extract(html)
    if text:
        print(text, end="")


if __name__ == "__main__":
    main()
