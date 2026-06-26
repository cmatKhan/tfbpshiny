"""
SQL generators for the ``topn_results`` comparison table.

Adapted from ``modules/comparison/queries.py::topn_responsive_ratio``.
Functions return SQL strings (no side effects) so they can be called from a
Jupyter notebook to inspect the query before running the full pipeline.
The coordinator is the only code that calls ``.execute()``.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Per-dataset configuration (self-contained copy, not imported from modules/)
# ---------------------------------------------------------------------------

#: callingcards target locus tags excluded from top-N (matching R analysis)
CC_TARGET_BLACKLIST: tuple[str, ...] = (
    "YOR201C",
    "YOR202W",
    "YOR203W",
    "YCL018W",
    "YEL021W",
)

#: Harbison dedup CTE: aggregate to one row per (binding_sample, regulator, target)
#: keeping the minimum p-value, restricted to YPD condition.
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

#: Per-binding-dataset kwargs for top-N analysis.
#: Keys: binding_sample_col, rank_col, rank_asc, target_blacklist, binding_dedup_cte.
BINDING_TOPN_CONFIGS: dict[str, dict[str, Any]] = {
    "callingcards": dict(
        binding_sample_col="sample_id",
        rank_col="poisson_pval",
        rank_asc=True,
        target_blacklist=CC_TARGET_BLACKLIST,
        binding_dedup_cte="",
    ),
    "callingcards_mindel": dict(
        binding_sample_col="sample_id",
        rank_col="poisson_pval",
        rank_asc=True,
        target_blacklist=CC_TARGET_BLACKLIST,
        binding_dedup_cte="",
    ),
    "callingcards_500bp": dict(
        binding_sample_col="sample_id",
        rank_col="poisson_pval",
        rank_asc=True,
        target_blacklist=CC_TARGET_BLACKLIST,
        binding_dedup_cte="",
    ),
    "callingcards_intergenic": dict(
        binding_sample_col="sample_id",
        rank_col="poisson_pval",
        rank_asc=True,
        target_blacklist=CC_TARGET_BLACKLIST,
        binding_dedup_cte="",
    ),
    "harbison": dict(
        binding_sample_col="sample_id",
        rank_col="pvalue",
        rank_asc=True,
        target_blacklist=(),
        binding_dedup_cte=_HARBISON_DEDUP_CTE,
    ),
    "chec_m2025": dict(
        binding_sample_col="sample_id",
        rank_col="enrichment",
        rank_asc=False,
        target_blacklist=(),
        binding_dedup_cte="",
    ),
    "chec_m2025_mindel": dict(
        binding_sample_col="sample_id",
        rank_col="enrichment",
        rank_asc=False,
        target_blacklist=(),
        binding_dedup_cte="",
    ),
    "chec_m2025_500bp": dict(
        binding_sample_col="sample_id",
        rank_col="enrichment",
        rank_asc=False,
        target_blacklist=(),
        binding_dedup_cte="",
    ),
    "chec_m2025_intergenic": dict(
        binding_sample_col="sample_id",
        rank_col="enrichment",
        rank_asc=False,
        target_blacklist=(),
        binding_dedup_cte="",
    ),
    "chec_m2025_peaks": dict(
        binding_sample_col="sample_id",
        rank_col="peak_score",
        rank_asc=False,
        target_blacklist=(),
        binding_dedup_cte="",
    ),
    "rossi": dict(
        binding_sample_col="sample_id",
        rank_col="enrichment",
        rank_asc=False,
        target_blacklist=(),
        binding_dedup_cte="",
    ),
    "rossi_mindel": dict(
        binding_sample_col="sample_id",
        rank_col="enrichment",
        rank_asc=False,
        target_blacklist=(),
        binding_dedup_cte="",
    ),
    "rossi_500bp": dict(
        binding_sample_col="sample_id",
        rank_col="enrichment",
        rank_asc=False,
        target_blacklist=(),
        binding_dedup_cte="",
    ),
    "rossi_intergenic": dict(
        binding_sample_col="sample_id",
        rank_col="enrichment",
        rank_asc=False,
        target_blacklist=(),
        binding_dedup_cte="",
    ),
    "rossi_peaks": dict(
        binding_sample_col="sample_id",
        rank_col="peak_score",
        rank_asc=False,
        target_blacklist=(),
        binding_dedup_cte="",
    ),
}

#: Perturbation datasets eligible for top-N analysis (no per-dataset kwargs needed).
PERTURBATION_TOPN_DATASETS: frozenset[str] = frozenset(
    {
        "hackett",
        "hughes_overexpression",
        "hughes_knockout",
        "hu_reimand",
        "kemmeren",
        "degron",
    }
)

#: Map: perturbation db_name → (effect_col, pvalue_col).
#: Empty string means the column does not exist.
PERTURBATION_DATASET_COLUMNS: dict[str, tuple[str, str]] = {
    "degron": ("log2FoldChange", "padj"),
    "hughes_overexpression": ("mean_norm_log2fc", ""),
    "hughes_knockout": ("mean_norm_log2fc", ""),
    "kemmeren": ("Madj", "pval"),
    "hackett": ("log2_shrunken_timecourses", ""),
    "hu_reimand": ("effect", "pval"),
}


# ---------------------------------------------------------------------------
# SQL generators
# ---------------------------------------------------------------------------


def topn_schema_sql() -> str:
    """
    Return the ``CREATE TABLE topn_results`` DDL (empty — no data).

    :returns: ``CREATE TABLE topn_results (…)`` SQL string.
    :rtype: str

    """
    return """
CREATE TABLE topn_results (
    binding_source_sample       VARCHAR  NOT NULL,
    perturbation_source_sample  VARCHAR  NOT NULL,
    regulator_locus_tag         VARCHAR  NOT NULL,
    top_n                       INTEGER  NOT NULL,
    rank_col                    VARCHAR  NOT NULL,
    rank_asc                    BOOLEAN  NOT NULL,
    effect_threshold            DOUBLE   NOT NULL,
    pvalue_threshold            DOUBLE   NOT NULL,
    n                           INTEGER  NOT NULL,
    n_responsive                INTEGER  NOT NULL,
    responsive_ratio            DOUBLE   NOT NULL,
    PRIMARY KEY (
        binding_source_sample,
        perturbation_source_sample,
        regulator_locus_tag,
        top_n, rank_col, rank_asc,
        effect_threshold, pvalue_threshold
    )
);
"""


def _responsive_expr(
    perturbation_view: str,
    effect_threshold: float,
    pvalue_threshold: float,
    param_prefix: str,
    params: dict[str, Any],
) -> str:
    """
    Build a SQL expression evaluating to 1 (responsive) or 0.

    :param perturbation_view: Dataset name (key in ``PERTURBATION_DATASET_COLUMNS``).
    :param effect_threshold: Absolute effect magnitude must exceed this.
    :param pvalue_threshold: P-value must be below this (ignored when no pvalue col).
    :param param_prefix: Namespace prefix for SQL parameter names.
    :param params: Dict populated in-place with threshold values.
    :returns: SQL CASE expression string evaluating to 1 or 0.
    :rtype: str

    """
    cols = PERTURBATION_DATASET_COLUMNS.get(perturbation_view, ("", ""))
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
        return "CAST(p.responsive AS INTEGER)"


def topn_pair_select_sql(
    binding_view: str,
    binding_hf_repo: str,
    binding_hf_config: str,
    perturbation_view: str,
    pert_hf_repo: str,
    pert_hf_config: str,
    binding_sample_col: str,
    rank_col: str,
    rank_asc: bool,
    target_blacklist: tuple[str, ...],
    binding_dedup_cte: str,
    top_n: int,
    effect_threshold: float,
    pvalue_threshold: float,
    regulator_subset: tuple[str, ...] = (),
    param_prefix: str = "p",
) -> tuple[str, dict[str, Any]]:
    """
    Return a SELECT that produces ``topn_results``-shaped rows for one pair.

    The SELECT includes composite ``source_sample`` IDs, analysis parameters,
    and result columns — ready for the coordinator to wrap in
    ``INSERT INTO topn_results``.  Adapted from
    ``modules/comparison/queries.py::topn_responsive_ratio``.

    No user-level filters are applied; the query covers all samples in both
    datasets (subject to harbison YPD dedup when applicable).

    :param binding_view: Binding dataset name (DuckDB view/table name).
    :param binding_hf_repo: HuggingFace repo for the binding dataset.
    :param binding_hf_config: HuggingFace config for the binding dataset.
    :param perturbation_view: Perturbation dataset name.
    :param pert_hf_repo: HuggingFace repo for the perturbation dataset.
    :param pert_hf_config: HuggingFace config for the perturbation dataset.
    :param binding_sample_col: Column in the binding view for sample identifier.
    :param rank_col: Column used to rank binding hits.
    :param rank_asc: If ``True``, lower values rank better (p-values).
    :param target_blacklist: Target locus tags excluded from ranking.
    :param binding_dedup_cte: Optional CTE body SQL replacing the default
        binding SELECT (used for Harbison YPD dedup).
    :param top_n: Number of top binding targets to keep per binding sample.
    :param effect_threshold: Minimum absolute effect size to count as responsive.
    :param pvalue_threshold: Maximum p-value to count as responsive.
    :param regulator_subset: If non-empty, restrict both CTEs to these
        ``regulator_locus_tag`` values (for chunked execution).
    :param param_prefix: Namespace prefix for SQL parameters.
    :returns: ``(sql, params)`` tuple.
    :rtype: tuple[str, dict]

    """
    params: dict[str, Any] = {}
    rank_dir = "ASC" if rank_asc else "DESC"

    reg_in_clause = ""
    if regulator_subset:
        reg_ph = ", ".join(
            f"$reg_{param_prefix}_{i}" for i in range(len(regulator_subset))
        )
        for i, reg in enumerate(regulator_subset):
            params[f"reg_{param_prefix}_{i}"] = reg
        reg_in_clause = f"regulator_locus_tag IN ({reg_ph})"

    # Build the binding CTE body.
    if binding_dedup_cte:
        binding_cte_body = binding_dedup_cte
    else:
        blacklist_clauses: list[str] = []
        if target_blacklist:
            ph = ", ".join(
                f"$bl_{param_prefix}_{i}" for i in range(len(target_blacklist))
            )
            blacklist_clauses.append(f"target_locus_tag NOT IN ({ph})")
            for i, tag in enumerate(target_blacklist):
                params[f"bl_{param_prefix}_{i}"] = tag
        binding_extra = (
            "WHERE " + " AND ".join(blacklist_clauses) if blacklist_clauses else ""
        )
        binding_cte_body = f"""
        SELECT
            CAST({binding_sample_col} AS VARCHAR) AS binding_sample_id,
            regulator_locus_tag,
            target_locus_tag,
            {rank_col}
        FROM {binding_view}
        {binding_extra}
        """

    if reg_in_clause:
        binding_cte_body = (
            f"SELECT * FROM ({binding_cte_body}) AS _binding_batch"
            f" WHERE {reg_in_clause}"
        )

    # Perturbation responsive expression (uses $params for thresholds).
    responsive_expr = _responsive_expr(
        perturbation_view,
        effect_threshold,
        pvalue_threshold,
        param_prefix,
        params,
    )

    pert_clauses: list[str] = []
    if reg_in_clause:
        pert_clauses.append(f"p.{reg_in_clause}")
    pert_filter_where = f"WHERE {' AND '.join(pert_clauses)}" if pert_clauses else ""

    top_n_key = f"{param_prefix}_top_n"
    params[top_n_key] = top_n

    # Escaped literal strings for composite ID construction (no user input).
    b_prefix = f"{binding_hf_repo};{binding_hf_config};".replace("'", "''")
    p_prefix = f"{pert_hf_repo};{pert_hf_config};".replace("'", "''")

    # Literal values for the analysis-parameter columns.
    rank_col_safe = rank_col.replace("'", "''")
    rank_asc_sql = "TRUE" if rank_asc else "FALSE"

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
    ),
    summary AS (
        SELECT
            b.binding_sample_id,
            b.regulator_locus_tag,
            pert.perturbation_sample_id,
            COUNT(*)                               AS n,
            SUM(pert.is_responsive)::INTEGER       AS n_responsive,
            SUM(pert.is_responsive)::DOUBLE / COUNT(*) AS responsive_ratio
        FROM top_n_binding b
        JOIN perturbation pert
            ON  b.regulator_locus_tag = pert.regulator_locus_tag
            AND b.target_locus_tag    = pert.target_locus_tag
        GROUP BY b.binding_sample_id, b.regulator_locus_tag, pert.perturbation_sample_id
    )
    SELECT
        '{b_prefix}' || binding_sample_id         AS binding_source_sample,
        '{p_prefix}' || perturbation_sample_id    AS perturbation_source_sample,
        regulator_locus_tag,
        ${top_n_key}::INTEGER                     AS top_n,
        '{rank_col_safe}'                         AS rank_col,
        {rank_asc_sql}                            AS rank_asc,
        {effect_threshold!r}::DOUBLE              AS effect_threshold,
        {pvalue_threshold!r}::DOUBLE              AS pvalue_threshold,
        n,
        n_responsive,
        responsive_ratio
    FROM summary
    """
    return sql, params
