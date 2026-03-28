# Copyright 2023 Janek Bevendorff
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import errno
import hashlib
import logging
import os
import subprocess
import tempfile

from extraction_benchmark.paths import THIRD_PARTY_PATH

_logger = logging.getLogger("wceb.web2text")

WEB2TEXT_BASEPATH = os.path.join(THIRD_PARTY_PATH, "web2text")
WEB2TEXT_PYTHONPATH = os.path.join(WEB2TEXT_BASEPATH, "src", "main", "python")
WEB2TEXT_VENV = os.path.join(WEB2TEXT_BASEPATH, "venv")
WEB2TEXT_JAR = os.path.join(THIRD_PARTY_PATH, "web2text.jar")


def _venv_python() -> str:
    for name in ("python3.7", "python3", "python"):
        p = os.path.join(WEB2TEXT_VENV, "bin", name)
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(
        errno.ENOENT,
        os.path.join(WEB2TEXT_VENV, "bin"),
        "Нет интерпретатора в venv Web2Text. Создайте venv по README: "
        "cd third-party/web2text && python3.7 -m venv venv && pip install ...",
    )


def _resolve_java_home() -> str | None:
    """Java 8 для Scala/JAR; приоритет WCEB_JAVA_HOME, JAVA_HOME, типичные пути Debian/Ubuntu."""
    for key in ("WCEB_JAVA_HOME", "JAVA_HOME"):
        v = os.environ.get(key, "").strip()
        if v and os.path.isdir(v):
            return v
    for candidate in (
        "/usr/lib/jvm/java-8-openjdk-amd64",
        "/usr/lib/jvm/java-1.8.0-openjdk-amd64",
    ):
        if os.path.isdir(candidate):
            return candidate
    return None


if not os.path.isdir(WEB2TEXT_PYTHONPATH):
    raise FileNotFoundError(
        errno.ENOENT,
        WEB2TEXT_BASEPATH,
        "Не найден Web2Text. Выполните: git submodule update --init --recursive",
    )

if not os.path.isdir(WEB2TEXT_VENV):
    raise FileNotFoundError(
        errno.ENOENT,
        WEB2TEXT_VENV,
        "Нет venv Web2Text. См. README: third-party/web2text, python3.7 -m venv venv && pip install ...",
    )

if not os.path.isfile(WEB2TEXT_JAR):
    raise FileNotFoundError(
        errno.ENOENT,
        WEB2TEXT_JAR,
        "Ожидается fat-jar third-party/web2text.jar (см. third-party/web2text/README.md).",
    )


def extract(html: str) -> str:
    """
    Pipeline: ExtractPageFeatures (Scala) → classify (TF 1.x в venv) → ApplyLabelsToPage (Scala).
    Второй аргумент Scala — output *basename*; файлы: {basename}_block_features.csv и _edge_features.csv.
    Python main.py classify ожидает тот же basename (см. main.py: file_base + '_block_features.csv').
    """
    scala_cmd = ["scala", "-cp", WEB2TEXT_JAR]
    py = _venv_python()
    main_py = os.path.join(WEB2TEXT_PYTHONPATH, "main.py")
    hash_id = hashlib.sha256(html.encode("utf-8")).hexdigest()

    proc_env = os.environ.copy()
    proc_env["VIRTUAL_ENV"] = WEB2TEXT_VENV
    proc_env["PATH"] = f"{WEB2TEXT_VENV}/bin:{proc_env.get('PATH', '')}"
    java_home = _resolve_java_home()
    if java_home:
        proc_env["JAVA_HOME"] = java_home

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Один префикс для всех артефактов во временной директории
        file_base = os.path.join(tmp_dir, hash_id)
        html_file = file_base + ".html"
        labels_file = file_base + ".labels"
        text_file = file_base + ".txt"

        with open(html_file, "w", encoding="utf-8") as f:
            f.write(html)

        def _run_step(name: str, cmd: list[str]) -> None:
            _logger.info("Web2Text: %s", name)
            p = subprocess.run(
                cmd,
                env=proc_env,
                cwd=WEB2TEXT_PYTHONPATH,
                capture_output=True,
                text=True,
            )
            if p.returncode != 0:
                _logger.warning("Web2Text %s stderr: %s", name, (p.stderr or "")[:4000])
                _logger.warning("Web2Text %s stdout: %s", name, (p.stdout or "")[:2000])
                raise RuntimeError(f"Web2Text: {name} failed (exit {p.returncode})")

        # (1) Scala пишет {file_base}_block_features.csv и {file_base}_edge_features.csv
        _run_step(
            "ExtractPageFeatures",
            scala_cmd + ["ch.ethz.dalab.web2text.ExtractPageFeatures", html_file, file_base],
        )

        # (2) python main.py classify <basename> <labels_out> — НЕ путь к .features
        _run_step(
            "classify",
            [py, main_py, "classify", file_base, labels_file],
        )

        # (3) метки — CSV из шага 2
        _run_step(
            "ApplyLabelsToPage",
            scala_cmd + ["ch.ethz.dalab.web2text.ApplyLabelsToPage", html_file, labels_file, text_file],
        )

        with open(text_file, encoding="utf-8", errors="replace") as f:
            return f.read()
