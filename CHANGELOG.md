# Changelog

All notable changes to this extended version of the Web Content Extraction Benchmark will be documented in this file.

## [1.0.0] - Extended Version

### Added

- **BLEU metric evaluation**: Added BLEU score as a new evaluation metric alongside ROUGE and Levenshtein
  - Implementation in `src/extraction_benchmark/eval.py`
  - Support for BLEU in aggregation and visualization
  - Added to CLI commands: `wceb eval score bleu` and `wceb eval aggregate bleu`

- **Crawl4AI integration**: Added Crawl4AI as a new extraction model
  - Implementation in `src/extraction_benchmark/extractors/extractors.py`
  - Uses WebScrapingStrategy and DefaultMarkdownGenerator with PruningContentFilter
  - Integrated into the benchmark evaluation pipeline

- **Extended datasets**: Added two new datasets to the benchmark
  - Canola dataset
  - Newspaper3k dataset
  - Both datasets integrated into the evaluation system

### Changed

- Updated `src/extraction_benchmark/globals.py` to include:
  - Crawl4AI in the model list
  - BLEU in the scores list
  - Canola and Newspaper3k in the dataset mapping

- Updated evaluation commands to support BLEU metric
- Updated documentation in README.md

### Based on

This extended version is based on:
- Original Web Content Extraction Benchmark v1.0.0
- Repository: https://github.com/chatnoir-eu/web-content-extraction-benchmark
- Paper: "An Empirical Comparison of Web Content Extraction Algorithms" (Bevendorff et al., SIGIR 2023)

### License

This project maintains the Apache License 2.0 from the original project.

