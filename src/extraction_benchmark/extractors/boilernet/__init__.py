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
import logging
import os
import subprocess
import tempfile

from extraction_benchmark.paths import THIRD_PARTY_PATH

_logger = logging.getLogger("wceb.boilernet")

BOILERNET_THIRD_PARTY_PATH = os.path.join(THIRD_PARTY_PATH, "boilernet-tf1")
BOILERNET_VENV_PATH = os.path.join(BOILERNET_THIRD_PARTY_PATH, "venv")
BOILERNET_RUNNER = os.path.join(BOILERNET_THIRD_PARTY_PATH, "run_boilernet.py")


def _venv_python() -> str:
    for candidate in ("python3.7", "python3", "python"):
        python_path = os.path.join(BOILERNET_VENV_PATH, "bin", candidate)
        if os.path.isfile(python_path):
            return python_path
    raise FileNotFoundError(
        errno.ENOENT,
        BOILERNET_VENV_PATH,
        "Для Boilernet создайте venv с TensorFlow 1.15: "
        "python3.7 -m venv third-party/boilernet-tf1/venv && "
        "source third-party/boilernet-tf1/venv/bin/activate && "
        "pip install tensorflow==1.15.0 beautifulsoup4==4.12.3 html5lib==1.1 nltk==3.8.1 numpy==1.24.2",
    )


def extract(html: str) -> str:
    if not os.path.isdir(BOILERNET_VENV_PATH):
        raise FileNotFoundError(
            errno.ENOENT,
            BOILERNET_VENV_PATH,
            "Boilernet venv отсутствует. См. SETUP.md — раздел Boilernet (TensorFlow 1.15).",
        )
    if not os.path.isfile(BOILERNET_RUNNER):
        raise FileNotFoundError(
            errno.ENOENT,
            BOILERNET_RUNNER,
            "Отсутствует скрипт run_boilernet.py в third-party/boilernet-tf1.",
        )

    python_bin = _venv_python()

    with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False, encoding="utf-8") as html_file:
        html_file.write(html)
        html_path = html_file.name

    proc_env = os.environ.copy()
    proc_env["VIRTUAL_ENV"] = BOILERNET_VENV_PATH
    proc_env["PATH"] = os.path.join(BOILERNET_VENV_PATH, "bin") + ":" + proc_env.get("PATH", "")

    try:
        proc = subprocess.run(
            [python_bin, BOILERNET_RUNNER, html_path],
            env=proc_env,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            _logger.warning("Boilernet subprocess stderr:\n%s", (proc.stderr or "").strip())
            _logger.warning("Boilernet subprocess stdout:\n%s", (proc.stdout or "").strip())
            return ""

        return (proc.stdout or "").strip()
    finally:
        try:
            os.remove(html_path)
        except OSError:
            pass
