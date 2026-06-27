# flake8: noqa
"""SQL queries for the Comparison (DTO / Top-N by Binding) module."""

from __future__ import annotations

import os
from typing import Any

import pandas as pd
from labretriever import VirtualDB

from tfbpshiny.modules.perturbation.queries import DATASET_COLUMNS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: callingcards target locus tags excluded from the top-N analysis (matching R)
CC_TARGET_BLACKLIST = ("YOR201C", "YOR202W", "YOR203W", "YCL018W", "YEL021W")

#: Pseudo-value added before -log10 to avoid log(0)
DTO_LOG_PSEUDO = 1e-3

#: Default top-N cutoff
DEFAULT_TOP_N = 25


#: Default regulators-per-chunk for the comparison top-N executor. Each chunked
#: query touches only its regulator slice of the multi-million-row views, which
#: bounds the DuckDB working set (and thus the connection's retained memory
#: high-water mark) across a session. Overridable via the
#: ``COMPARISON_REGULATORS_PER_CHUNK`` environment variable; set it to ``0`` to
#: disable chunking (whole-pair execution).
_DEFAULT_REGULATORS_PER_CHUNK = 400


def _comparison_regulators_per_chunk() -> int | None:
    """
    Number of regulators to process per chunk in :func:`topn_all_pairs_sql`.

    Read from the ``COMPARISON_REGULATORS_PER_CHUNK`` environment variable at
    call time (so it can be set after import); defaults to
    :data:`_DEFAULT_REGULATORS_PER_CHUNK` when unset or invalid. A positive value
    subdivides each pair into regulator batches of that size to bound peak (and
    retained) memory at the cost of more, smaller queries. Set the env var to
    ``0`` (or a negative value) to disable chunking and run each pair whole.

    :returns: Positive batch size, or ``None`` for whole-pair execution.
    :rtype: int | None

    """
    raw = os.environ.get("COMPARISON_REGULATORS_PER_CHUNK")
    if raw is None or raw == "":
        return _DEFAULT_REGULATORS_PER_CHUNK
    try:
        n = int(raw)
    except ValueError:
        return _DEFAULT_REGULATORS_PER_CHUNK
    return n if n > 0 else None


# ---------------------------------------------------------------------------
# DTO query
# ---------------------------------------------------------------------------

_DTO_SQL = """
SELECT
    d.binding_id_source,
    d.perturbation_id_source,
    d.dto_empirical_pvalue,
    d.dto_fdr,
    d.binding_set_size,
    d.perturbation_set_size,
    CAST(d.binding_id_id   AS VARCHAR)    AS binding_sample_id,
    CAST(d.perturbation_id_id AS VARCHAR) AS pert_sample_id,
    COALESCE(CAST(h.time AS VARCHAR), 'standard') AS time
FROM dto_expanded d
LEFT JOIN (
    SELECT DISTINCT sample_id, time FROM hackett_meta WHERE time = 45
) h
    ON  d.perturbation_id_source = 'hackett'
    AND CAST(d.perturbation_id_id AS VARCHAR) = CAST(h.sample_id AS VARCHAR)
LEFT JOIN (
    SELECT DISTINCT sample_id FROM callingcards
) cc
    ON  d.binding_id_source = 'callingcards'
    AND CAST(d.binding_id_id AS VARCHAR) = CAST(cc.sample_id AS VARCHAR)
LEFT JOIN (
    SELECT DISTINCT sample_id FROM harbison WHERE condition = 'YPD'
) harb
    ON  d.binding_id_source = 'harbison'
    AND CAST(d.binding_id_id AS VARCHAR) = CAST(harb.sample_id AS VARCHAR)
WHERE
    d.pr_ranking_column = 'log2fc'
    AND (d.perturbation_id_source != 'hackett'     OR h.sample_id IS NOT NULL)
    AND (d.binding_id_source      != 'callingcards' OR cc.sample_id IS NOT NULL)
    AND (d.binding_id_source      != 'harbison'     OR harb.sample_id IS NOT NULL)
"""


def fetch_dto_data(
    vdb: VirtualDB, sql_only: bool = False
) -> pd.DataFrame | tuple[str, dict]:
    """
    Fetch DTO empirical p-value data from ``dto_expanded``.

    Requires ``hackett_analysis_set`` to be registered first (done by
    :func:`tfbpshiny.utils.vdb_init.initialize_data`).

    :param vdb: VirtualDB instance.
    :param sql_only: If ``True`` return ``(sql, {})`` instead of executing.
    :returns: DataFrame with columns ``binding_id_source``,
        ``perturbation_id_source``, ``dto_empirical_pvalue``, ``dto_fdr``,
        ``binding_set_size``, ``perturbation_set_size``, ``binding_sample_id``,
        ``pert_sample_id``, ``time``.

    """
    if sql_only:
        return _DTO_SQL, {}
    return vdb.query(_DTO_SQL)


# ---------------------------------------------------------------------------
# Top-N responsive ratio query
# ---------------------------------------------------------------------------

_HARBISON_DEDUP_CTE = """
    SELECT
        CAST(sample_id AS VARCHAR) AS binding_sample_id,
        regulator_locus_tag,
        target_locus_tag,
        MIN(pvalue) AS pvalue
    FROM harbison
    WHERE sample_id IN (
        SELECT sample_id FROM harbison_meta WHERE condition = 'YPD'
    )
    GROUP BY sample_id, regulator_locus_tag, target_locus_tag
"""


def _build_where(clauses: list[str]) -> str:
    return ("WHERE " + " AND ".join(clauses)) if clauses else ""


def _build_filter_where(
    filters: dict[str, Any] | None,
    params: dict[str, Any],
    prefix: str,
) -> str:
    """Build a WHERE clause from a dataset_filters spec, populating params in-place."""
    if not filters:
        return ""
    clauses: list[str] = []
    for field, spec in filters.items():
        kind = spec["type"]
        val = spec["value"]
        p = f"{prefix}_{field}".replace(" ", "_")
        if kind == "categorical":
            placeholders = ", ".join(f"$cat_{p}_{i}" for i in range(len(val)))
            clauses.append(f'"{field}" IN ({placeholders})')
            for i, v in enumerate(val):
                params[f"cat_{p}_{i}"] = v
        elif kind == "numeric":
            clauses.append(f'"{field}" BETWEEN $num_{p}_lo AND $num_{p}_hi')
            params[f"num_{p}_lo"] = val[0]
            params[f"num_{p}_hi"] = val[1]
        elif kind == "bool":
            clauses.append(f'"{field}" = $bool_{p}')
            params[f"bool_{p}"] = bool(val)
    return _build_where(clauses)


def _meta_sample_filter(
    view: str,
    filters: dict[str, Any] | None,
    params: dict[str, Any],
    prefix: str,
) -> str:
    """
    Build a bare ``sample_id IN (meta subquery)`` predicate resolving filters via
    metadata.

    Dataset filters target metadata columns that need not be carried on the data
    view. The filter resolves to a ``sample_id`` set against ``{view}_meta`` so the
    data-view scan only needs its projected columns plus a ``sample_id`` membership
    test. Returns a bare clause (no leading ``WHERE``) for composition with
    :func:`_build_where`; empty string when there are no filters.

    :param view: Data view name; its meta view is ``{view}_meta``.
    :param filters: Filter spec for the dataset, or ``None``.
    :param params: Dict populated in-place with bound parameter values.
    :param prefix: Namespace prefix for parameter names.
    :returns: ``"sample_id IN (...)"`` clause, or ``""``.

    """
    inner = _build_filter_where(filters, params, prefix)
    if not inner:
        return ""
    return f"sample_id IN (SELECT sample_id FROM {view}_meta {inner})"


def _responsive_expr(
    perturbation_view: str,
    effect_threshold: float,
    pvalue_threshold: float,
    param_prefix: str,
    params: dict[str, Any],
) -> str:
    """
    Build a SQL expression that evaluates to 1 (responsive) or 0.

    Uses the effect and pvalue columns from ``DATASET_COLUMNS`` for the given
    perturbation view. If the dataset has no pvalue column only the effect
    threshold is applied.

    :param perturbation_view: Dataset name (key in ``DATASET_COLUMNS``).
    :param effect_threshold: Absolute effect magnitude must exceed this.
    :param pvalue_threshold: P-value must be below this (ignored if no pvalue
        column exists for the dataset).
    :param param_prefix: Namespace prefix for SQL parameter names.
    :param params: Dict populated in-place with threshold parameter values.
    :returns: SQL CASE expression string evaluating to 1 or 0.

    """
    cols = DATASET_COLUMNS.get(perturbation_view, ("", ""))
    effect_col, pvalue_col = cols[0], cols[1]

    eff_key = f"{param_prefix}_eff_thresh"
    pval_key = f"{param_prefix}_pval_thresh"
    params[eff_key] = effect_threshold

    if effect_col and pvalue_col:
        params[pval_key] = pvalue_threshold
        return (
            f"CASE WHEN ABS(p.{effect_col}) > ${eff_key} "
            f"AND p.{pvalue_col} < ${pval_key} THEN 1 ELSE 0 END"
        )
    elif effect_col:
        return f"CASE WHEN ABS(p.{effect_col}) > ${eff_key} THEN 1 ELSE 0 END"
    else:
        # Fall back to pre-computed responsive column
        return "CAST(p.responsive AS INTEGER)"


def topn_responsive_ratio(
    vdb: VirtualDB,
    binding_view: str,
    perturbation_view: str,
    binding_sample_col: str,
    rank_col: str,
    top_n: int = DEFAULT_TOP_N,
    effect_threshold: float = 0.0,
    pvalue_threshold: float = 0.05,
    binding_filters: dict[str, Any] | None = None,
    perturbation_filters: dict[str, Any] | None = None,
    rank_asc: bool = True,
    target_blacklist: tuple[str, ...] = (),
    binding_dedup_cte: str = "",
    regulator_subset: tuple[str, ...] = (),
    param_prefix: str = "p",
    sql_only: bool = False,
) -> pd.DataFrame | tuple[str, dict]:
    """
    Compute the top-N-by-binding responsive ratio for one (binding, perturbation) pair.

    Computes the intersection of targets present in both datasets first, then
    ranks only the shared targets per binding sample (PARTITION BY
    binding_sample_id) and keeps the top ``top_n``.  This ensures that the
    top-N slots are not consumed by binding targets that have no corresponding
    perturbation measurement.  Responsiveness is evaluated dynamically using
    the effect/pvalue thresholds from ``_responsive_expr``.

    :param vdb: VirtualDB instance.
    :param binding_view: View name for binding data.
    :param perturbation_view: View name for perturbation data.
    :param binding_sample_col: Column in binding view for the sample identifier.
    :param rank_col: Column used to rank binding hits.
    :param top_n: Number of top binding targets to keep per binding sample.
    :param effect_threshold: Minimum absolute effect size to count as responsive.
    :param pvalue_threshold: Maximum p-value to count as responsive (ignored if
        the dataset has no p-value column).
    :param binding_filters: dataset_filters spec for the binding dataset.
    :param perturbation_filters: dataset_filters spec for the perturbation dataset.
    :param rank_asc: If ``True``, lower values of ``rank_col`` rank better.
    :param target_blacklist: Locus tags to exclude from binding targets.
    :param binding_dedup_cte: Optional CTE body SQL to replace the default
        binding SELECT (used for Harbison dedup).
    :param regulator_subset: If non-empty, restrict both the binding and
        perturbation CTEs to these ``regulator_locus_tag`` values. Used to
        compute one regulator batch at a time so peak memory stays bounded; the
        per-regulator results are independent, so batching is lossless.
    :param param_prefix: Namespace prefix for SQL parameters to avoid collisions.
    :param sql_only: If ``True`` return ``(sql, params)`` instead of executing.

    """
    params: dict[str, Any] = {}
    rank_dir = "ASC" if rank_asc else "DESC"

    # Optional regulator-batch restriction (applied to both CTEs).
    reg_in_clause = ""
    if regulator_subset:
        reg_ph = ", ".join(
            f"$reg_{param_prefix}_{i}" for i in range(len(regulator_subset))
        )
        for i, reg in enumerate(regulator_subset):
            params[f"reg_{param_prefix}_{i}"] = reg
        reg_in_clause = f"regulator_locus_tag IN ({reg_ph})"

    # binding CTE
    if binding_dedup_cte:
        binding_cte_body = binding_dedup_cte
    else:
        b_sample_filter = _meta_sample_filter(
            binding_view, binding_filters, params, prefix=f"{param_prefix}_b"
        )
        blacklist_clauses = []
        if b_sample_filter:
            blacklist_clauses.append(b_sample_filter)
        if target_blacklist:
            ph = ", ".join(
                f"$bl_{param_prefix}_{i}" for i in range(len(target_blacklist))
            )
            blacklist_clauses.append(f"target_locus_tag NOT IN ({ph})")
            for i, tag in enumerate(target_blacklist):
                params[f"bl_{param_prefix}_{i}"] = tag
        binding_extra = _build_where(blacklist_clauses)
        binding_cte_body = f"""
        SELECT
            CAST({binding_sample_col} AS VARCHAR) AS binding_sample_id,
            regulator_locus_tag,
            target_locus_tag,
            {rank_col}
        FROM {binding_view}
        {binding_extra}
        """

    # Restrict the binding CTE (normal or dedup) to the regulator batch, if any.
    if reg_in_clause:
        binding_cte_body = (
            f"SELECT * FROM ({binding_cte_body}) AS _binding_batch"
            f" WHERE {reg_in_clause}"
        )

    # perturbation responsive expression
    responsive_expr = _responsive_expr(
        perturbation_view,
        effect_threshold,
        pvalue_threshold,
        param_prefix,
        params,
    )

    # perturbation CTE: filter resolves to a sample_id set via the meta view.
    # The perturbation table is aliased ``p``, so qualify the membership test.
    pert_sample_filter = _meta_sample_filter(
        perturbation_view, perturbation_filters, params, prefix=f"{param_prefix}_p"
    )
    pert_clauses: list[str] = []
    if pert_sample_filter:
        pert_clauses.append(f"p.{pert_sample_filter}")
    if reg_in_clause:
        pert_clauses.append(f"p.{reg_in_clause}")
    pert_filter_where = f"WHERE {' AND '.join(pert_clauses)}" if pert_clauses else ""

    top_n_key = f"{param_prefix}_top_n"
    params[top_n_key] = top_n

    sql = f"""
    WITH binding AS (
        {binding_cte_body}
    ),
    perturbation AS (
        SELECT
            CAST(p.sample_id AS VARCHAR) AS perturbation_sample_id,
            p.regulator_locus_tag,
            p.target_locus_tag,
            {responsive_expr} AS is_responsive
        FROM {perturbation_view} p
        {pert_filter_where}
    ),
    intersecting_targets AS (
        SELECT DISTINCT b.regulator_locus_tag, b.target_locus_tag
        FROM binding b
        INNER JOIN perturbation pert
            ON  b.regulator_locus_tag = pert.regulator_locus_tag
            AND b.target_locus_tag    = pert.target_locus_tag
    ),
    binding_ranked AS (
        SELECT
            b.binding_sample_id,
            b.regulator_locus_tag,
            b.target_locus_tag,
            b.{rank_col},
            RANK() OVER (
                PARTITION BY b.binding_sample_id
                ORDER BY b.{rank_col} {rank_dir}
            ) AS rnk
        FROM binding b
        INNER JOIN intersecting_targets it
            ON  b.regulator_locus_tag = it.regulator_locus_tag
            AND b.target_locus_tag    = it.target_locus_tag
        WHERE b.regulator_locus_tag != b.target_locus_tag
    ),
    top_n_binding AS (
        SELECT binding_sample_id, regulator_locus_tag, target_locus_tag
        FROM binding_ranked
        WHERE rnk <= ${top_n_key}
    )
    SELECT
        b.binding_sample_id,
        b.regulator_locus_tag,
        pert.perturbation_sample_id,
        COUNT(*)                                         AS n,
        SUM(pert.is_responsive)::INTEGER                 AS n_responsive,
        SUM(pert.is_responsive)::DOUBLE / COUNT(*)       AS responsive_ratio
    FROM top_n_binding b
    JOIN perturbation pert
        ON  b.regulator_locus_tag = pert.regulator_locus_tag
        AND b.target_locus_tag    = pert.target_locus_tag
    GROUP BY b.binding_sample_id, b.regulator_locus_tag, pert.perturbation_sample_id
    """

    if sql_only:
        return sql, params
    return vdb.query(sql, **params)


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

#: Per-binding-source kwargs passed to topn_responsive_ratio (excluding filters).
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
        binding_dedup_cte=_HARBISON_DEDUP_CTE,
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

#: Per-perturbation-source kwargs passed to topn_responsive_ratio (excluding filters).
PERTURBATION_CONFIGS: dict[str, dict] = {
    "hackett": {},
    "hughes_overexpression": {},
    "hughes_knockout": {},
    "hu_reimand": {},
    "kemmeren": {},
    "degron": {},
}


def _binding_regulators(
    vdb: VirtualDB, binding_view: str, binding_filters: dict[str, Any] | None
) -> list[str]:
    """
    Return the sorted distinct regulator locus tags for a binding dataset.

    Resolved from the small ``{binding_view}_meta`` view (post-filter), so it is
    cheap. Used to subdivide a pair into regulator batches for chunked execution.

    :param vdb: VirtualDB instance.
    :param binding_view: Binding dataset name.
    :param binding_filters: Active filter spec for the binding dataset.
    :returns: Sorted list of distinct regulator locus tags.
    :rtype: list[str]

    """
    params: dict[str, Any] = {}
    where = _build_filter_where(binding_filters, params, prefix="rl")
    sql = f"SELECT DISTINCT regulator_locus_tag FROM {binding_view}_meta {where}"
    df = vdb.query(sql, **params)
    return sorted(t for t in df["regulator_locus_tag"].dropna().tolist())


def topn_all_pairs_sql(
    vdb: VirtualDB,
    pairs: list[tuple[str, str]],
    filters: dict[str, Any],
    top_n: int,
    preset: dict[str, tuple[float, float]],
) -> pd.DataFrame:
    """
    Compute the top-N responsive-ratio summary for all (binding, perturbation) pairs.

    Executes **one pair at a time** (never a single UNION ALL across pairs) and,
    when ``COMPARISON_REGULATORS_PER_CHUNK`` is set, subdivides each pair into
    regulator batches. Each query's intermediates are released before the next
    runs, so peak memory stays bounded by a single pair/batch rather than the
    whole grid. The small per-(sample, regulator) summary rows are accumulated in
    Python; this is the same shape the matrix and box-plot distributions consume.

    Responsiveness thresholds are looked up per perturbation dataset from
    ``preset`` (``"*"`` is the fallback key).

    :param vdb: VirtualDB instance.
    :param pairs: List of ``(binding_db, perturbation_db)`` tuples.
    :param filters: Active filter dict keyed by dataset name.
    :param top_n: Number of top binding targets per binding sample.
    :param preset: Per-dataset responsiveness thresholds; see
        :data:`~tfbpshiny.utils.vdb_init.DEFAULT_RESPONSIVENESS_PRESETS`.
    :returns: DataFrame with all columns returned by ``topn_responsive_ratio``
        plus ``pair_key`` (``"{b_db}__{p_db}"``).

    """
    if not pairs:
        return pd.DataFrame()

    chunk = _comparison_regulators_per_chunk()
    frames: list[pd.DataFrame] = []

    for b_db, p_db in pairs:
        b_cfg = BINDING_CONFIGS.get(b_db)
        p_cfg = PERTURBATION_CONFIGS.get(p_db)
        if b_cfg is None or p_cfg is None:
            continue
        effect_threshold, pvalue_threshold = preset.get(
            p_db, preset.get("*", (0.0, 0.05))
        )

        # Regulator batches for this pair: one empty batch (= whole pair) when
        # chunking is disabled, otherwise size-``chunk`` slices of the binding
        # dataset's regulators.
        if chunk:
            regulators = _binding_regulators(vdb, b_db, filters.get(b_db))
            batches: list[tuple[str, ...]] = [
                tuple(regulators[i : i + chunk])
                for i in range(0, len(regulators), chunk)
            ] or [()]
        else:
            batches = [()]

        pair_key = f"{b_db}__{p_db}"
        for batch in batches:
            pair_sql, pair_params = topn_responsive_ratio(
                vdb=vdb,
                binding_view=b_db,
                perturbation_view=p_db,
                top_n=top_n,
                effect_threshold=effect_threshold,
                pvalue_threshold=pvalue_threshold,
                binding_filters=filters.get(b_db),
                perturbation_filters=filters.get(p_db),
                regulator_subset=batch,
                param_prefix="bp",
                sql_only=True,
                **b_cfg,
                **p_cfg,
            )
            assert isinstance(pair_sql, str) and isinstance(pair_params, dict)
            df = vdb.query(pair_sql, **pair_params)
            if not df.empty:
                df["pair_key"] = pair_key
                frames.append(df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
