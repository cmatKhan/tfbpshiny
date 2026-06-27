"""Binding dataset column configuration and correlation query helpers for Phase 2."""

from __future__ import annotations

from tfbpshiny.utils.corr_query import fetch_corr_pairs, get_filtered_sample_ids  # noqa: F401

# Maps binding db_name -> (score_col, pvalue_col) — must match what was materialized.
BINDING_DATASET_COLUMNS: dict[str, tuple[str, str]] = {
    "callingcards": ("callingcards_enrichment", "poisson_pval"),
    "callingcards_mindel": ("callingcards_enrichment", "poisson_pval"),
    "callingcards_500bp": ("callingcards_enrichment", "poisson_pval"),
    "callingcards_intergenic": ("callingcards_enrichment", "poisson_pval"),
    "harbison": ("effect", "pvalue"),
    "rossi": ("enrichment", "poisson_pval"),
    "rossi_mindel": ("enrichment", "poisson_pval"),
    "rossi_500bp": ("enrichment", "poisson_pval"),
    "rossi_intergenic": ("enrichment", "poisson_pval"),
    "chec_m2025": ("enrichment", "poisson_pval"),
    "chec_m2025_mindel": ("enrichment", "poisson_pval"),
    "chec_m2025_500bp": ("enrichment", "poisson_pval"),
    "chec_m2025_intergenic": ("enrichment", "poisson_pval"),
}

# Keep DATASET_COLUMNS for backward compatibility with any remaining callers.
DATASET_COLUMNS = BINDING_DATASET_COLUMNS


__all__ = [
    "BINDING_DATASET_COLUMNS",
    "DATASET_COLUMNS",
    "fetch_corr_pairs",
    "get_filtered_sample_ids",
]
