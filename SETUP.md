# Настройка проекта

## Быстрый старт (уже выполнено)

1. **Зависимости** — `poetry install` (venv в `.venv/`)
2. **Датасеты** — данные скопированы из `добавить позже/` в `datasets/combined/`
3. **Проверка** — `poetry run wceb --help`

## Полная настройка с нуля

### 1. Установка зависимостей

```bash
poetry config virtualenvs.in-project true
poetry install
poetry shell   # опционально: активировать venv
```

### 2. Датасеты

**Вариант A: Git LFS (полные данные)**

```bash
sudo apt install git-lfs
git lfs install
git lfs pull
cd datasets && tar xf combined.tar.xz && cd ..
```

**Вариант B: Локальные данные**

Скопируйте `ground-truth/` и `html/` в `datasets/combined/`:

```bash
mkdir -p datasets/combined
cp -r "добавить позже/ground-truth" datasets/combined/
cp -r "добавить позже/html" datasets/combined/
```

### 3. Dragnet и Extractnet (опционально)

**Простое решение:** использовать готовые wheel из `third-party/`:

```bash
PYVER=$(poetry run python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
poetry run pip install --no-deps third-party/dragnet-2.0.4-cp${PYVER/./}-cp${PYVER/./}-linux_x86_64.whl third-party/extractnet-2.0.7-cp${PYVER/./}-cp${PYVER/./}-linux_x86_64.whl
poetry run pip install "ftfy>=4.1.0,<5.0.0" cchardet onnxruntime sklearn-crfsuite==0.3.6
```

Или для Python 3.11:
```bash
poetry run pip install --no-deps third-party/dragnet-2.0.4-cp311-cp311-linux_x86_64.whl third-party/extractnet-2.0.7-cp311-cp311-linux_x86_64.whl
poetry run pip install "ftfy>=4.1.0,<5.0.0" cchardet onnxruntime sklearn-crfsuite==0.3.6
```

### 4. Web2Text (опционально, отдельный venv)

Требует: **Java 8**, **Scala**, **Python 3.7**.

**Автоматическая настройка:**
```bash
chmod +x scripts/setup-web2text.sh
./scripts/setup-web2text.sh
```

**Ручная настройка (по README):**
```bash
# Java 8 и Scala
sudo apt install openjdk-8-jdk scala

# Submodule
git submodule update --init --recursive

# Python venv (как в README)
cd third-party/web2text
python3.7 -m venv venv
source venv/bin/activate
pip install numpy==1.18.0 tensorflow==1.15.0 tensorflow-gpu==1.15.0 protobuf==3.20.1 future==0.18.3
deactivate
cd ../..
```

Запуск Web2Text: `poetry run wceb extract -m web2text -d data-ml-test -p 1`

### 5. Использование

```bash
# Извлечение (один экстрактор, один датасет)
poetry run wceb extract -m readability -d data-ml-test -p 1

# Оценка
poetry run wceb eval score levenshtein
poetry run wceb eval score bleu
poetry run wceb eval score rouge
```

## Изменения в pyproject.toml

Для совместимости с Crawl4AI внесены правки:
- `python = ">=3.10,<3.12"` (crawl4ai требует Python ≥3.10)
- `lxml = ">=5.3,<6.0"` (crawl4ai требует lxml ≥5.3)
