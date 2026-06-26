"""
SQL generators for the ``correlations`` comparison table.

Adapted from ``modules/binding/queries.py::_corr_pair_sql_impl``.
Functions return SQL strings (no side effects) so they can be called from a
Jupyter notebook to inspect the query before running the full pipeline.
The coordinator is the only code that calls ``.execute()``.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Per-dataset measurement columns (self-contained copy, not imported from modules/)
# ---------------------------------------------------------------------------

#: Binding dataset → (effect_col, pvalue_col).
#: Empty string means the column does not exist in that dataset.
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

#: Perturbation dataset → (effect_col, pvalue_col).
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


def correlations_schema_sql() -> str:
    """
    Return the ``CREATE TABLE correlations`` DDL (empty — no data).

    :returns: ``CREATE TABLE correlations (…)`` SQL string.
    :rtype: str

    """
    return """
CREATE TABLE correlations (
    source_sample_a      VARCHAR  NOT NULL,
    source_sample_b      VARCHAR  NOT NULL,
    regulator_locus_tag  VARCHAR  NOT NULL,
    comparison_type      VARCHAR  NOT NULL,
    method               VARCHAR  NOT NULL,
    score_col_a          VARCHAR  NOT NULL,
    score_col_b          VARCHAR  NOT NULL,
    correlation          DOUBLE   NOT NULL,
    n_shared_targets     INTEGER  NOT NULL,
    PRIMARY KEY (
        source_sample_a, source_sample_b,
        regulator_locus_tag,
        method, score_col_a, score_col_b
    )
);
"""


def correlation_pair_select_sql(
    view_a: str,
    hf_repo_a: str,
    hf_config_a: str,
    col_a: str,
    view_b: str,
    hf_repo_b: str,
    hf_config_b: str,
    col_b: str,
    method: str,
    comparison_type: str,
    param_prefix: str = "p",
) -> tuple[str, dict[str, Any]]:
    """
    Return a SELECT that produces ``correlations``-shaped rows for one dataset pair.

    The SELECT includes composite ``source_sample`` IDs, comparison metadata,
    and correlation values — ready for the coordinator to wrap in
    ``INSERT INTO correlations``.  Adapted from
    ``modules/binding/queries.py::_corr_pair_sql_impl``.

    Pair ordering is enforced by the caller: pass datasets in lexicographic
    order (``view_a <= view_b``) so each unordered pair is stored exactly once.
    No user-level filters are applied; the query covers all samples.

    :param view_a: First dataset name (DuckDB view/table name); must be <= view_b.
    :param hf_repo_a: HuggingFace repo for dataset A.
    :param hf_config_a: HuggingFace config for dataset A.
    :param col_a: Measurement column to use from dataset A.
    :param view_b: Second dataset name; must be >= view_a.
    :param hf_repo_b: HuggingFace repo for dataset B.
    :param hf_config_b: HuggingFace config for dataset B.
    :param col_b: Measurement column to use from dataset B.
    :param method: ``'pearson'`` or ``'spearman'``.
    :param comparison_type: ``'binding'`` or ``'perturbation'``.
    :param param_prefix: Namespace prefix for SQL parameters.
    :returns: ``(sql, params)`` tuple.
    :rtype: tuple[str, dict]

    """
    params: dict[str, Any] = {}

    is_pvalue_a = "pval" in col_a.lower()
    is_pvalue_b = "pval" in col_b.lower()
    order_a = f"{col_a} ASC" if is_pvalue_a else f"ABS({col_a}) DESC"
    order_b = f"{col_b} ASC" if is_pvalue_b else f"ABS({col_b}) DESC"

    prefix_a_str = f"{hf_repo_a};{hf_config_a};".replace("'", "''")
    prefix_b_str = f"{hf_repo_b};{hf_config_b};".replace("'", "''")
    col_a_safe = col_a.replace("'", "''")
    col_b_safe = col_b.replace("'", "''")
    comparison_type_safe = comparison_type.replace("'", "''")
    method_safe = method.replace("'", "''")

    if method == "spearman":
        sql = f"""
WITH
  a_raw AS (
    SELECT regulator_locus_tag, target_locus_tag,
           CAST(sample_id AS VARCHAR) AS sample_id,
           {col_a}
    FROM {view_a}
    WHERE {col_a} IS NOT NULL
      AND NOT isinf({col_a})
      AND NOT isnan({col_a})
  ),
  b_raw AS (
    SELECT regulator_locus_tag, target_locus_tag,
           CAST(sample_id AS VARCHAR) AS sample_id,
           {col_b}
    FROM {view_b}
    WHERE {col_b} IS NOT NULL
      AND NOT isinf({col_b})
      AND NOT isnan({col_b})
  ),
  joined AS (
    SELECT
      a_raw.regulator_locus_tag,
      a_raw.sample_id  AS id_a,
      b_raw.sample_id  AS id_b,
      a_raw.{col_a}    AS val_a,
      b_raw.{col_b}    AS val_b
    FROM a_raw
    INNER JOIN b_raw
      ON  a_raw.regulator_locus_tag = b_raw.regulator_locus_tag
     AND a_raw.target_locus_tag    = b_raw.target_locus_tag
  ),
  ranked AS (
    SELECT
      regulator_locus_tag,
      id_a, id_b,
      RANK() OVER (
        PARTITION BY regulator_locus_tag, id_a, id_b
        ORDER BY {order_a.replace(col_a, 'val_a')}
      ) AS rank_a,
      RANK() OVER (
        PARTITION BY regulator_locus_tag, id_a, id_b
        ORDER BY {order_b.replace(col_b, 'val_b')}
      ) AS rank_b
    FROM joined
  ),
  agg AS (
    SELECT
      regulator_locus_tag,
      id_a, id_b,
      corr(rank_a, rank_b) AS correlation,
      COUNT(*)             AS n_shared_targets
    FROM ranked
    GROUP BY regulator_locus_tag, id_a, id_b
    HAVING COUNT(*) >= 3
  )
SELECT
  '{prefix_a_str}' || id_a  AS source_sample_a,
  '{prefix_b_str}' || id_b  AS source_sample_b,
  regulator_locus_tag,
  '{comparison_type_safe}'   AS comparison_type,
  '{method_safe}'            AS method,
  '{col_a_safe}'             AS score_col_a,
  '{col_b_safe}'             AS score_col_b,
  correlation,
  n_shared_targets::INTEGER
FROM agg
WHERE correlation IS NOT NULL AND NOT isnan(correlation)
"""
    else:
        # pearson
        sql = f"""
WITH
  a_raw AS (
    SELECT regulator_locus_tag, target_locus_tag,
           CAST(sample_id AS VARCHAR) AS sample_id,
           {col_a}
    FROM {view_a}
    WHERE {col_a} IS NOT NULL
      AND NOT isinf({col_a})
      AND NOT isnan({col_a})
  ),
  b_raw AS (
    SELECT regulator_locus_tag, target_locus_tag,
           CAST(sample_id AS VARCHAR) AS sample_id,
           {col_b}
    FROM {view_b}
    WHERE {col_b} IS NOT NULL
      AND NOT isinf({col_b})
      AND NOT isnan({col_b})
  ),
  agg AS (
    SELECT
      a_raw.regulator_locus_tag,
      a_raw.sample_id                     AS id_a,
      b_raw.sample_id                     AS id_b,
      corr(a_raw.{col_a}, b_raw.{col_b}) AS correlation,
      COUNT(*)                            AS n_shared_targets
    FROM a_raw
    INNER JOIN b_raw
      ON  a_raw.regulator_locus_tag = b_raw.regulator_locus_tag
     AND a_raw.target_locus_tag    = b_raw.target_locus_tag
    GROUP BY a_raw.regulator_locus_tag, a_raw.sample_id, b_raw.sample_id
    HAVING COUNT(*) >= 3
  )
SELECT
  '{prefix_a_str}' || id_a  AS source_sample_a,
  '{prefix_b_str}' || id_b  AS source_sample_b,
  regulator_locus_tag,
  '{comparison_type_safe}'   AS comparison_type,
  '{method_safe}'            AS method,
  '{col_a_safe}'             AS score_col_a,
  '{col_b_safe}'             AS score_col_b,
  correlation,
  n_shared_targets::INTEGER
FROM agg
WHERE correlation IS NOT NULL
"""

    return sql, params
