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

import re


def extract_bs4(html, **_):
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, 'html.parser')
    for e in soup(['script', 'style', 'noscript']):
        e.decompose()
    return soup.get_text(separator=' ', strip=True)


def extract_boilerpipe(html, **_):
    import boilerpipe.extract as boilerpipe
    text = boilerpipe.Extractor(extractor='ArticleExtractor', html=html)
    text = text.getText()
    return str(text)


def extract_xpath_text(html, **_):
    import lxml.html
    root = lxml.html.fromstring(html)
    text = ' '.join(root.xpath('//body[1]//*[not(name()="script") and not(name()="style")]/text()'))
    text = re.sub(r'(\s+\n\s*)', '\n', text)
    return re.sub(r'[ \t]{2,}', ' ', text)


def extract_news_please(html, **_):
    import newsplease
    return newsplease.NewsPlease.from_html(html, url=None).maintext


def extract_readability(html, **_):
    import readability, html_text
    doc = readability.Document(html)
    text = html_text.extract_text(doc.summary(html_partial=True))
    return text


def extract_go_domdistiller(html, **_):
    from extraction_benchmark.extractors import go_domdistiller
    return go_domdistiller.extract(html)


def extract_inscriptis(html, **_):
    import inscriptis
    text = inscriptis.get_text(html)
    return text


def extract_html_text(html, **_):
    import html_text
    return html_text.extract_text(html)


def extract_resiliparse(html, **_):
    from resiliparse.extract import html2text
    from resiliparse.parse.html import HTMLTree
    return html2text.extract_plain_text(HTMLTree.parse(html),
                                        preserve_formatting=True,
                                        main_content=True,
                                        list_bullets=False,
                                        comments=False,
                                        links=False,
                                        alt_texts=False)


def extract_bte(html, **_):
    from extraction_benchmark.extractors import bte
    return bte.html2text(html)


def extract_trafilatura(html, **_):
    import trafilatura
    return trafilatura.extract(html, include_comments=False)


def extract_justext(html, **_):
    import justext
    article = ' '.join(
        [p.text for p in justext.justext(html, justext.get_stoplist("English"), 50, 200, 0.1, 0.2, 0.2, 200, True)
         if not p.is_boilerplate])
    return article


def extract_goose3(html, **_):
    from goose3 import Goose, configuration
    c = configuration.Configuration()
    c.http_timeout = 5

    with Goose(c) as g:
        article = g.extract(raw_html=html)
        return article.cleaned_text


def extract_lxml_cleaner(html, **_):
    from bs4 import BeautifulSoup
    from lxml.html.clean import Cleaner

    tag_blacklist = [
        # important
        'aside', 'embed', 'footer', 'form', 'head', 'iframe', 'menu', 'object', 'script',
        # other content
        'applet', 'audio', 'canvas', 'figure', 'map', 'picture', 'svg', 'video',
        # secondary
        'area', 'blink', 'button', 'datalist', 'dialog',
        'frame', 'frameset', 'fieldset', 'link', 'input', 'ins', 'label', 'legend',
        'marquee', 'math', 'menuitem', 'nav', 'noscript', 'optgroup', 'option',
        'output', 'param', 'progress', 'rp', 'rt', 'rtc', 'select', 'source',
        'style', 'track', 'template', 'textarea', 'time', 'use',
    ]

    cleaner = Cleaner(
        annoying_tags=False,  # True
        comments=True,
        embedded=False,  # True
        forms=True,  # True
        frames=True,  # True
        javascript=True,
        links=False,
        meta=False,
        page_structure=False,
        processing_instructions=True,
        remove_unknown_tags=False,
        safe_attrs_only=False,
        scripts=True,
        style=False,
        kill_tags=tag_blacklist
    )
    return BeautifulSoup(cleaner.clean_html(html), 'html.parser').get_text(separator=' ', strip=True)


def extract_boilernet(html, **_):
    from extraction_benchmark.extractors import boilernet
    return boilernet.extract(html)


def extract_web2text(html, **_):
    from extraction_benchmark.extractors import web2text
    return web2text.extract(html)


def extract_newspaper3k(html, **_):
    import newspaper
    article = newspaper.Article('')
    article.set_html(html)
    article.parse()
    return article.text


def extract_dragnet(html, **_):
    from dragnet import extract_content
    return extract_content(html, encoding='utf8')


def extract_extractnet(html, **_):
    from extractnet import Extractor
    return Extractor().extract(html, encoding='utf8').get('content', '')


def extract_crawl4ai(html, **kwargs):
    """
    Извлечение основного контента с помощью Crawl4AI.
    Использует WebScrapingStrategy + DefaultMarkdownGenerator + PruningContentFilter
    """
    from crawl4ai.content_scraping_strategy import WebScrapingStrategy
    from crawl4ai.content_filter_strategy import PruningContentFilter
    from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
    import re

    def markdown_to_text(md: str) -> str:
        """Преобразование Markdown в чистый текст"""
        if not md:
            return ""
        # Убираем markdown разметку
        md = re.sub(r'#+\s*', '', md)                               # заголовки
        md = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', md)            # ссылки
        md = re.sub(r'!\[([^\]]*)\]\([^)]+\)', '', md)              # картинки
        md = re.sub(r'\*\*([^*]+)\*\*', r'\1', md)                  # **жирный**
        md = re.sub(r'\*([^*]+)\*', r'\1', md)                      # *курсив*
        md = re.sub(r'-{3,}', '', md)                               # разделители
        md = re.sub(r'^\s*>+\s*', '', md, flags=re.MULTILINE)       # цитаты
        md = re.sub(r'\n\s*\n', '\n\n', md)                         # переносы
        return md.strip()

    try:
        # Создаем фильтр контента
        content_filter = PruningContentFilter(
            threshold=0.5,  # Более мягкий порог
            threshold_type="fixed",
            min_word_threshold=5  # Минимум 5 слов
        )

        # Создаем генератор markdown с фильтром
        md_generator = DefaultMarkdownGenerator(content_filter=content_filter)

        # Генерируем markdown
        markdown_result = md_generator.generate_markdown(
            input_html=html,
            base_url="",
            content_filter=content_filter
        )

        if markdown_result and markdown_result.fit_markdown:
            text = markdown_to_text(markdown_result.fit_markdown)
            
            # Дополнительная очистка
            text = re.sub(r'\s+', ' ', text)  # Убираем лишние пробелы
            text = text.strip()
            
            return text
        else:
            return ""

    except Exception as e:
        # Fallback: используем только WebScrapingStrategy
        try:
            scraper = WebScrapingStrategy()
            result = scraper.scrap(
                url="",
                html=html,
                word_count_threshold=5,
                excluded_tags=[
                    "nav", "footer", "aside", "header", "menu", "menuitem",
                    "script", "style", "noscript", "link", "meta",
                    "form", "button", "input", "select", "textarea", "label",
                    "iframe", "embed", "object", "applet", "canvas", "svg", "video", "audio",
                    "ins", "advertisement", "ads", "banner", "promo",
                    "social", "share", "follow", "like", "tweet",
                    "sidebar", "widget", "toolbar", "breadcrumb", "pagination",
                    "comment", "comments", "related", "recommended", "popular",
                    "tag", "tags", "category", "categories", "archive"
                ],
                exclude_external_links=True,
                exclude_external_images=True,
                only_text=True
            )
            
            if result.success and result.cleaned_html:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(result.cleaned_html, 'html.parser')
                return soup.get_text(separator=' ', strip=True)
            else:
                return ""
        except:
            return ""


def extract_data_ml_lr(html, page_id, **_):
    """
    Экстрактор на основе LogisticRegression (data-ml признаки).
    """
    try:
        from extraction_benchmark.data_ml_models import extract_with_model
    except ImportError:
        return ""
    return extract_with_model(html, page_id, "lr")


def extract_data_ml_rf(html, page_id, **_):
    """
    Экстрактор на основе RandomForest (data-ml признаки).
    """
    try:
        from extraction_benchmark.data_ml_models import extract_with_model
    except ImportError:
        return ""
    return extract_with_model(html, page_id, "rf")


def extract_data_ml_catboost(html, page_id, **_):
    """
    Экстрактор на основе CatBoost (data-ml признаки).
    """
    try:
        from extraction_benchmark.data_ml_models import extract_with_model
    except ImportError:
        return ""
    return extract_with_model(html, page_id, "catboost")


def _get_ensemble_model_list(best_only=False, weighted=False):
    def _ls():
        if best_only or weighted:
            return [
                (extract_goose3, 2 if weighted else 1),
                (extract_readability, 2 if weighted else 1),
                (extract_trafilatura, 2 if weighted else 1),
                (extract_go_domdistiller, 1),
                (extract_resiliparse, 1),
                (extract_web2text, 1),
                (extract_newspaper3k, 1),
                (extract_dragnet, 1),
                (extract_boilernet, 1),
                (extract_bte, 1),
                (extract_justext, 1),
                (extract_boilerpipe, 1),
            ]

        return [(m, 1) for m in list_extractors(names_only=False, include_ensembles=False)]

    return zip(*[(m.__name__.replace('extract_', ''), w) for m, w in _ls()])


def extract_ensemble_majority(html, page_id):
    from extraction_benchmark.extractors import ensemble
    models, weights = _get_ensemble_model_list()
    return ensemble.extract_majority_vote(html, page_id, models, weights, int(len(models) * .66))


def extract_ensemble_best(html, page_id):
    from extraction_benchmark.extractors import ensemble
    models, weights = _get_ensemble_model_list(best_only=True)
    return ensemble.extract_majority_vote(html, page_id, models, weights, int(len(models) * .66))


def extract_ensemble_weighted(html, page_id):
    from extraction_benchmark.extractors import ensemble
    models, weights = _get_ensemble_model_list(best_only=True, weighted=True)
    return ensemble.extract_majority_vote(html, page_id, models, weights, int(len(models) * .66))


def list_extractors(names_only=True, include_ensembles=False):
    """
    Get a list of all supported extraction systems.

    :param names_only: only return a list of strings (otherwise return extractor routines)
    :param include_ensembles: include ensemble extractors in the list
    :return: list of extractor names or functions
    """
    return [(n.replace('extract_', '') if names_only else m) for n, m in globals().items()
            if n.startswith('extract_') and (not n.startswith('extract_ensemble') or include_ensembles)]
