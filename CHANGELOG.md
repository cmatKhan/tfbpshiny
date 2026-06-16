# Changelog

All notable changes to this project will be documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.1.0] - 2026-06-15

### Changed

- Deferred the startup data materialization to a background task so the app becomes
  interactive in a few seconds instead of blocking ~30-57s on every cold start.
  Data-querying tabs show an "optimizing" banner and unlock automatically once the
  background materialization completes; queries are gated until then to keep the
  shared DuckDB connection single-threaded.
- Made startup loading banners report accurate cold-start times (up to ~10s for
  initial load, up to ~50s for the background optimization step).
- Moved production (EC2/Docker) and shinyapps.io deployment instructions out of the
  README into `docs/development.md`; the default log level is now `WARNING`.

### Removed

- "Under development" banner from the Home page.

### Fixed

- Diagonal cell sample count in the dataset matrix now reports the total row count
  rather than the distinct sample count.
- Corrected the end-to-end navigation test selectors to match the current UI.
- Packaging fix so `configure_logger` resolves when installed from PyPI/GitHub.

### Updated

- labretriever updated to 1.1.3, which is on bioconda.

---

## [1.0.0] - 2026-06-12

### Added

- Initial public release of TFBPShiny.
- Dashboard interface for exploring transcription factor binding and perturbation
  data from the Brent Lab yeast collection.
- Dataset Selection module with filter controls for binding and perturbation datasets.
- Binding module with correlation and scatter visualizations.
- Perturbation module with correlation and scatter visualizations.
- Comparison module with three subtabs: Compare Datasets (binding vs. perturbation
  matrix), Compare Promoter Definitions (enrichment scores across four promoter sets:
  Kang, Mindel, 500bp, Intergenic), and Compare Analysis Methods (promoter enrichment
  vs. original peaks for ChIP-exo and ChEC-seq datasets).
- `python -m tfbpshiny launch` CLI entry point: downloads the HuggingFace dataset
  cache on first run and serves the app on subsequent runs from the same directory.
  Supports `--cache-dir`, `--skip-initialize`, `--no-materialize`, `--port`, `--host`,
  and `--debug` flags.
- Projected in-memory materialization of dataset views at startup for improved query
  performance; disabled via `--no-materialize` or `TFBPSHINY_MATERIALIZE=0`.
- Docker Compose production stack with Traefik reverse proxy and AWS CloudWatch
  logging.
- shinyapps.io deployment support via `shinyapps_entry.py`.
- Terraform configuration for EC2 provisioning.
