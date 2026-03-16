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

from contextlib import redirect_stderr, redirect_stdout
from functools import partial
import io
from itertools import product
import json
import logging
from multiprocessing import get_context
import os
from typing import Any, Dict
import warnings
import time

import click

from extraction_benchmark.dataset_readers import read_datasets, read_raw_dataset
from extraction_benchmark.extractors import extractors
from extraction_benchmark.paths import *


def _dict_to_jsonl(filepath, lines_dict: Dict[str, Any]):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        for page_id in sorted(lines_dict):
            json.dump({'page_id': page_id, **lines_dict[page_id]}, f, indent=None, ensure_ascii=False)
            f.write('\n')


def extract_ground_truth(datasets):
    """
    Convert ground truth from raw dataset to JSON format.

    :param datasets: list of input dataset
    :return: set of page IDs that were extracted
    """

    page_ids = set()
    for ds in datasets:
        with click.progressbar(read_raw_dataset(ds, True), label=f'Converting ground truth of {ds}') as ds_progress:
            extracted = {k: v for k, v in ds_progress}
            page_ids.update(extracted.keys())
        _dict_to_jsonl(os.path.join(DATASET_COMBINED_TRUTH_PATH, f'{ds}.jsonl'), extracted)
    return page_ids


def extract_raw_html(datasets, page_id_whitelist=None):
    """
    Convert HTML files from raw dataset to JSON format.

    :param datasets: list of input dataset
    :param page_id_whitelist: optional list of page IDs to include (if set, IDs not in this list will be skipped)
    """
    if page_id_whitelist and type(page_id_whitelist) is not set:
        page_id_whitelist = set(page_id_whitelist)

    for ds in datasets:
        out_dir = os.path.join(DATASET_COMBINED_HTML_PATH, ds)
        os.makedirs(out_dir, exist_ok=True)
        with click.progressbar(read_raw_dataset(ds, False), label=f'Converting HTML of {ds}') as ds_progress:
            for page_id, val in ds_progress:
                if page_id_whitelist and page_id not in page_id_whitelist:
                    continue
                if not val.get('html'):
                    continue
                with open(os.path.join(out_dir, page_id + '.html'), 'w') as f:
                    f.write(val['html'])


def _extract_with_model_expand_args(args, skip_existing=False, verbose=False):
    _extract_with_model(*args, skip_existing=skip_existing, verbose=verbose)


def _extract_with_model(model, dataset, skip_existing=False, verbose=False):
    model, model_name = model
    out_path = os.path.join(MODEL_OUTPUTS_PATH, dataset, model_name + '.jsonl')

    logger = logging.getLogger('wceb-extract')
    logger.setLevel(logging.INFO if verbose else logging.ERROR)
    
    # Также настраиваем логгер для crawl4ai
    crawl4ai_logger = logging.getLogger('crawl4ai_extractor')
    crawl4ai_logger.setLevel(logging.INFO if verbose else logging.WARNING)
    
    # Создаем обработчик для вывода в консоль, если его еще нет
    if not crawl4ai_logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(message)s')
        handler.setFormatter(formatter)
        crawl4ai_logger.addHandler(handler)

    extracted = {}
    if skip_existing and os.path.isfile(out_path):
        with open(out_path, 'r') as f:
            for line in f:
                j = json.loads(line)
                extracted[j['page_id']] = {k: v for k, v in j.items() if k != 'page_id'}

    # Подсчитываем общее количество файлов для обработки
    total_files = 0
    files_to_process = []
    for file_hash, in_data in read_datasets([dataset], False):
        if file_hash not in extracted:
            files_to_process.append((file_hash, in_data))
        total_files += 1
    
    processed_count = len(extracted)
    logger.info(f"[{model_name}] Датасет {dataset}: всего файлов {total_files}, уже обработано {processed_count}, осталось {len(files_to_process)}")

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        for i, (file_hash, in_data) in enumerate(files_to_process, 1):
            start_time = time.time()
            logger.info(f"[{model_name}] Обрабатываю файл {i}/{len(files_to_process)}: {file_hash}")

            out_data = dict(plaintext='', model=model_name)
            try:
                with redirect_stdout(io.StringIO()) as stdout, redirect_stderr(io.StringIO()) as stderr:
                    out_data['plaintext'] = model(in_data['html'], page_id=file_hash) or ''

                processing_time = time.time() - start_time
                text_length = len(out_data['plaintext'])
                logger.info(f"[{model_name}] Файл {file_hash} обработан за {processing_time:.2f}с, получено {text_length} символов текста")

                if stdout.getvalue():
                    logger.info(stdout.getvalue().strip())
                if stderr.getvalue():
                    logger.warning(stderr.getvalue().strip())
            except Exception as e:
                processing_time = time.time() - start_time
                logger.warning(f'[{model_name}] Ошибка при обработке {dataset} ({file_hash}) за {processing_time:.2f}с:')
                logger.warning(str(e))

            extracted[file_hash] = out_data
            
            # Сохраняем промежуточные результаты каждые 10 файлов
            # if i % 10 == 0:
            logger.info(f"[{model_name}] Сохраняю промежуточные результаты... ({i}/{len(files_to_process)})")
            _dict_to_jsonl(out_path, extracted)

    if not extracted:
        return

    logger.info(f"[{model_name}] Сохраняю финальные результаты в {out_path}")
    _dict_to_jsonl(out_path, extracted)


def extract(models, datasets, skip_existing, parallelism, verbose=False):
    """
    Extract datasets with the selected extraction models.

    :param models: list of extraction model names (if ``ground_truth == False``)
    :param datasets: list of dataset names under "datasets/raw"
    :param skip_existing: skip models for which an answer file exists already
    :param parallelism: number of parallel workers
    :param verbose: log error information
    """

    model = [(getattr(extractors, 'extract_' + m), m) for m in models]
    jobs = list(product(model, datasets))

    def item_show_func(j):
        if j:
            return f'Model: {j[0][1]}, Dataset: {j[1]}'

    if parallelism == 1:
        with click.progressbar(jobs, label='Running extrators', item_show_func=item_show_func) as progress:
            for job in progress:
                _extract_with_model_expand_args(job)
        return

    with get_context('spawn').Pool(processes=parallelism) as pool:
        try:
            with click.progressbar(pool.imap_unordered(partial(_extract_with_model_expand_args,
                                                               skip_existing=skip_existing, verbose=verbose), jobs),
                                   length=len(jobs), label='Running extrators') as progress:
                for _ in progress:
                    pass
        except KeyboardInterrupt:
            pool.terminate()

    click.echo(f'Model outputs written to {MODEL_OUTPUTS_PATH}')
