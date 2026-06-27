"""Perturbation dataset column configuration for Phase 2 DuckDB-based queries."""

from __future__ import annotations

from tfbpshiny.utils.corr_query import fetch_corr_pairs, get_filtered_sample_ids  # noqa: F401

# Maps perturbation db_name -> (effect_col, pvalue_col).
# Empty string means the column does not exist in that dataset.
DATASET_COLUMNS: dict[str, tuple[str, str]] = {
    "degron": ("log2FoldChange", "padj"),
    "hughes_overexpression": ("mean_norm_log2fc", ""),
    "hughes_knockout": ("mean_norm_log2fc", ""),
    "kemmeren": ("Madj", "pval"),
    "hackett": ("log2_shrunken_timecourses", ""),
    "hu_reimand": ("effect", "pval"),
}


__all__ = [
    "DATASET_COLUMNS",
    "fetch_corr_pairs",
    "get_filtered_sample_ids",
]
