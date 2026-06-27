# flake8: noqa
"""SQL queries for the Comparison (DTO / Top-N by Binding) module — Phase 2 DuckDB version."""

from __future__ import annotations

from typing import Any

import duckdb
import pandas as pd

from tfbpshiny.utils.corr_query import get_filtered_sample_ids

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: callingcards target locus tags excluded from the top-N analysis (matching R)
CC_TARGET_BLACKLIST = ("YOR201C", "YOR202W", "YOR203W", "YCL018W", "YEL021W")

#: Pseudo-value added before -log10 to avoid log(0)
DTO_LOG_PSEUDO = 1e-3

#: Default top-N cutoff
DEFAULT_TOP_N = 25

#: Default effect size threshold (|effect| must exceed this to be "responsive")
DEFAULT_EFFECT_THRESHOLD = 0.0

#: Default p-value threshold (pvalue must be below this to be "responsive")
DEFAULT_PVALUE_THRESHOLD = 0.05

# ---------------------------------------------------------------------------
# Source label maps (matching the R code)
# ---------------------------------------------------------------------------

# Promoter-set-aware constants -----------------------------------------------

#: Maps every binding db_name to its base label (promoter-set suffix stripped).
#: All promoter variants of the same dataset share the same base label.
BINDING_BASE_LABEL_MAP: dict[str, str] = {
    "callingcards": "2026 Calling Cards",
    "callingcards_mindel": "2026 Calling Cards",
    "callingcards_500bp": "2026 Calling Cards",
    "callingcards_intergenic": "2026 Calling Cards",
    "harbison": "2004 ChIP-chip",
    "rossi": "2021 ChIP-exo",
    "rossi_mindel": "2021 ChIP-exo",
    "rossi_500bp": "2021 ChIP-exo",
    "rossi_intergenic": "2021 ChIP-exo",
    "chec_m2025": "2025 ChEC-seq",
    "chec_m2025_mindel": "2025 ChEC-seq",
    "chec_m2025_500bp": "2025 ChEC-seq",
    "chec_m2025_intergenic": "2025 ChEC-seq",
}

#: Maps every binding db_name to its promoter-set label.
PROMOTER_SET_MAP: dict[str, str] = {
    "callingcards": "Kang",
    "callingcards_mindel": "Mindel",
    "callingcards_500bp": "500bp",
    "callingcards_intergenic": "Intergenic",
    "harbison": "Kang",
    "rossi": "Kang",
    "rossi_mindel": "Mindel",
    "rossi_500bp": "500bp",
    "rossi_intergenic": "Intergenic",
    "chec_m2025": "Kang",
    "chec_m2025_mindel": "Mindel",
    "chec_m2025_500bp": "500bp",
    "chec_m2025_intergenic": "Intergenic",
}

BINDING_LABEL_MAP: dict[str, str] = {
    "callingcards": "2026 Calling Cards",
    "harbison": "2004 ChIP-chip",
    "chec_m2025": "2025 ChEC-seq",
    "rossi": "2021 ChIP-exo",
    "chec_m2025_mindel": "2025 ChEC-seq (Mindel)",
    "rossi_mindel": "2021 ChIP-exo (Mindel)",
    "callingcards_mindel": "2026 Calling Cards (Mindel)",
    "rossi_500bp": "2021 ChIP-exo (500bp)",
    "chec_m2025_500bp": "2025 ChEC-seq (500bp)",
    "rossi_intergenic": "2021 ChIP-exo (Intergenic)",
    "chec_m2025_intergenic": "2025 ChEC-seq (Intergenic)",
    "callingcards_500bp": "2026 Calling Cards (500bp)",
    "callingcards_intergenic": "2026 Calling Cards (Intergenic)",
}

#: Maps primary binding db_name to an ordered list of its promoter-set variants,
#: in the same order as the promoter set selector choices: Kang, Mindel, 500bp, Intergenic.
#: The primary db_name itself is not included here; it represents the Kang variant.
PROMOTER_VARIANT_PAIRS: dict[str, list[str]] = {
    "rossi": ["rossi_mindel", "rossi_500bp", "rossi_intergenic"],
    "chec_m2025": ["chec_m2025_mindel", "chec_m2025_500bp", "chec_m2025_intergenic"],
    "callingcards": [
        "callingcards_mindel",
        "callingcards_500bp",
        "callingcards_intergenic",
    ],
}

# ---------------------------------------------------------------------------
# Method Comparison constants
# ---------------------------------------------------------------------------

#: Maps every binding db_name that appears in the Method Comparison tab to its
#: base label; all scoring variants of the same dataset share the same label.
METHOD_BASE_LABEL_MAP: dict[str, str] = {
    "chec_m2025": "2025 ChEC-seq",
    "chec_m2025_mindel": "2025 ChEC-seq",
    "chec_m2025_500bp": "2025 ChEC-seq",
    "chec_m2025_intergenic": "2025 ChEC-seq",
    "chec_m2025_peaks": "2025 ChEC-seq",
    "rossi": "2021 ChIP-exo",
    "rossi_mindel": "2021 ChIP-exo",
    "rossi_500bp": "2021 ChIP-exo",
    "rossi_intergenic": "2021 ChIP-exo",
    "rossi_peaks": "2021 ChIP-exo",
}

#: Human-readable label for each scoring variant in the Method Comparison tab.
SCORING_VARIANT_MAP: dict[str, str] = {
    "chec_m2025": "Promoter Enrichment (Kang)",
    "chec_m2025_mindel": "Promoter Enrichment (Mindel)",
    "chec_m2025_500bp": "Promoter Enrichment (500bp)",
    "chec_m2025_intergenic": "Promoter Enrichment (Intergenic)",
    "chec_m2025_peaks": "Original Peaks",
    "rossi": "Promoter Enrichment (Kang)",
    "rossi_mindel": "Promoter Enrichment (Mindel)",
    "rossi_500bp": "Promoter Enrichment (500bp)",
    "rossi_intergenic": "Promoter Enrichment (Intergenic)",
    "rossi_peaks": "Original Peaks",
}

#: Maps each primary binding dataset to the peaks variants produced by the
#: original authors' peak-calling pipeline.
PEAKS_VARIANT_MAP: dict[str, list[str]] = {
    "rossi": ["rossi_peaks"],
    "chec_m2025": ["chec_m2025_peaks"],
}

#: Display order for scoring variants within a subplot.
SCORING_VARIANT_ORDER: list[str] = [
    "Promoter Enrichment (Kang)",
    "Promoter Enrichment (Mindel)",
    "Promoter Enrichment (500bp)",
    "Promoter Enrichment (Intergenic)",
    "Original Peaks",
]

#: Color palette for scoring variants in the Method Comparison tab.
SCORING_VARIANT_COLORS: dict[str, str] = {
    "Promoter Enrichment (Kang)": "#4DBBD5",
    "Promoter Enrichment (Mindel)": "#00A087",
    "Promoter Enrichment (500bp)": "#7B4F9E",
    "Promoter Enrichment (Intergenic)": "#F39B7F",
    "Original Peaks": "#E64B35",
}

PERTURBATION_LABEL_MAP: dict[str, str] = {
    "hackett": "2020 Overexpression",
    "hughes_overexpression": "2006 Overexpression",
    "hughes_knockout": "2006 TFKO",
    "hu_reimand": "2007 TFKO",
    "kemmeren": "2014 TFKO",
    "degron": "2025 Degron",
}

# ---------------------------------------------------------------------------
# Per-source configuration for top-N analysis
# ---------------------------------------------------------------------------

#: Per-binding-source kwargs (informational — used by labeling helpers).
BINDING_CONFIGS: dict[str, dict] = {
    "callingcards": dict(
        binding_sample_col="sample_id",
        rank_col="poisson_pval",
        rank_asc=True,
        target_blacklist=CC_TARGET_BLACKLIST,
    ),
    "callingcards_mindel": dict(
        binding_sample_col="sample_id",
        rank_col="poisson_pval",
        rank_asc=True,
        target_blacklist=CC_TARGET_BLACKLIST,
    ),
    "callingcards_500bp": dict(
        binding_sample_col="sample_id",
        rank_col="poisson_pval",
        rank_asc=True,
        target_blacklist=CC_TARGET_BLACKLIST,
    ),
    "callingcards_intergenic": dict(
        binding_sample_col="sample_id",
        rank_col="poisson_pval",
        rank_asc=True,
        target_blacklist=CC_TARGET_BLACKLIST,
    ),
    "harbison": dict(
        binding_sample_col="sample_id",
        rank_col="pvalue",
        rank_asc=True,
    ),
    "chec_m2025": dict(
        binding_sample_col="sample_id",
        rank_col="enrichment",
        rank_asc=False,
    ),
    "rossi": dict(
        binding_sample_col="sample_id",
        rank_col="enrichment",
        rank_asc=False,
    ),
    "rossi_mindel": dict(
        binding_sample_col="sample_id",
        rank_col="enrichment",
        rank_asc=False,
    ),
    "rossi_500bp": dict(
        binding_sample_col="sample_id",
        rank_col="enrichment",
        rank_asc=False,
    ),
    "rossi_intergenic": dict(
        binding_sample_col="sample_id",
        rank_col="enrichment",
        rank_asc=False,
    ),
    "chec_m2025_mindel": dict(
        binding_sample_col="sample_id",
        rank_col="enrichment",
        rank_asc=False,
    ),
    "chec_m2025_500bp": dict(
        binding_sample_col="sample_id",
        rank_col="enrichment",
        rank_asc=False,
    ),
    "chec_m2025_intergenic": dict(
        binding_sample_col="sample_id",
        rank_col="enrichment",
        rank_asc=False,
    ),
    "chec_m2025_peaks": dict(
        binding_sample_col="sample_id",
        rank_col="peak_score",
        rank_asc=False,
    ),
    "rossi_peaks": dict(
        binding_sample_col="sample_id",
        rank_col="peak_score",
        rank_asc=False,
    ),
}

#: Per-perturbation-source kwargs (informational).
PERTURBATION_CONFIGS: dict[str, dict] = {
    "hackett": {},
    "hughes_overexpression": {},
    "hughes_knockout": {},
    "hu_reimand": {},
    "kemmeren": {},
    "degron": {},
}


# ---------------------------------------------------------------------------
# DuckDB-based top-N fetch
# ---------------------------------------------------------------------------


def fetch_topn_results(
    conn: duckdb.DuckDBPyConnection,
    pairs: list[tuple[str, str]],
    filters: dict[str, Any],
    top_n: int,
    preset: dict[str, tuple[float, float]],
) -> pd.DataFrame:
    """
    Fetch pre-computed topn_results for all (binding, perturbation) pairs.

    Reads from the pre-materialized ``topn_results`` table, filtering by the
    materialized top_n, effect_threshold, pvalue_threshold, and sample IDs derived
    from dataset-level filters via ``{db_name}_meta`` subqueries.

    :param conn: Read-only DuckDB connection.
    :param pairs: List of (binding_db, perturbation_db) tuples.
    :param filters: Active filter dict keyed by dataset name.
    :param top_n: Number of top binding targets per binding sample (must match
        what was materialized).
    :param preset: Per-dataset responsiveness thresholds; see
        :data:`~tfbpshiny.utils.vdb_init.DEFAULT_RESPONSIVENESS_PRESETS`.
    :returns: DataFrame with columns from topn_results plus ``pair_key``
        (``"{b_db}__{p_db}"``).

    """
    if not pairs:
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    for b_db, p_db in pairs:
        effect_threshold, pvalue_threshold = preset.get(p_db, preset.get("*", (0.0, 0.05)))
        try:
            row_b = conn.execute(
                "SELECT hf_repo, hf_config FROM dataset_registry WHERE db_name = ?",
                [b_db],
            ).df().iloc[0]
            row_p = conn.execute(
                "SELECT hf_repo, hf_config FROM dataset_registry WHERE db_name = ?",
                [p_db],
            ).df().iloc[0]
        except (IndexError, Exception):
            continue
        b_prefix = f"{row_b['hf_repo']};{row_b['hf_config']};"
        p_prefix = f"{row_p['hf_repo']};{row_p['hf_config']};"

        b_ids = get_filtered_sample_ids(conn, b_db, filters.get(b_db))
        p_ids = get_filtered_sample_ids(conn, p_db, filters.get(p_db))

        if not b_ids or not p_ids:
            continue

        phs_b = ", ".join(["?"] * len(b_ids))
        phs_p = ", ".join(["?"] * len(p_ids))
        sql = f"""
        SELECT
            regulator_locus_tag,
            split_part(binding_source_sample, ';', 3) AS binding_sample_id,
            split_part(perturbation_source_sample, ';', 3) AS perturbation_sample_id,
            n, n_responsive, responsive_ratio
        FROM topn_results
        WHERE top_n = ?
          AND effect_threshold = ?
          AND pvalue_threshold = ?
          AND binding_source_sample LIKE ?
          AND perturbation_source_sample LIKE ?
          AND split_part(binding_source_sample, ';', 3) IN ({phs_b})
          AND split_part(perturbation_source_sample, ';', 3) IN ({phs_p})
        """
        params: list[Any] = (
            [top_n, effect_threshold, pvalue_threshold, b_prefix + "%", p_prefix + "%"]
            + b_ids
            + p_ids
        )
        try:
            df = conn.execute(sql, params).df()
            if not df.empty:
                df["pair_key"] = f"{b_db}__{p_db}"
                frames.append(df)
        except Exception:
            pass

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def fetch_dto_data(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Fetch DTO empirical p-value data from the materialized dto table.

    :param conn: Read-only DuckDB connection.
    :returns: DataFrame with all columns from the ``dto`` table.

    """
    return conn.execute("SELECT * FROM dto").df()
