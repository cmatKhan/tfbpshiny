"""SQL query templates for the Perturbation analysis module."""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd
from labretriever import VirtualDB

from tfbpshiny.modules.binding.queries import _corr_pair_sql_impl

# Map of db_name -> (effect_col, pvalue_col, log10p_col, neglog10p_col).
# Empty string means the column does not exist in that dataset.
# log10p_col: precomputed log10(pval) column (positive, not yet negated).
# neglog10p_col: precomputed -log10(pval) column (already negated).
# TODO: this information should be moved to virtualdb config.
DATASET_COLUMNS: dict[str, tuple[str, str, str, str]] = {
    "degron": ("log2FoldChange", "padj", "", ""),
    "hughes_overexpression": ("mean_norm_log2fc", "", "", ""),
    "hughes_knockout": ("mean_norm_log2fc", "", "", ""),
    "kemmeren": ("Madj", "pval", "", ""),
    "hackett": ("log2_shrunken_timecourses", "", "", ""),
    "hu_reimand": ("effect", "pval", "", ""),
}

#: P-values below this floor are capped before -log10 is applied.
LOG10P_FLOOR = 1e-10


def get_measurement_column(
    db_name: str, which_measurement_field: Literal["effect", "pvalue", "log10pval"]
) -> str:
    """
    Return the column to use for a dataset given the user's preference.

    For ``"log10pval"``, returns the most direct available column:
    ``neglog10p_col`` > ``log10p_col`` > ``pvalue_col``. Falls back to the
    effect column when no p-value variant exists.

    For ``"pvalue"``, returns ``pvalue_col`` when non-empty, else ``effect_col``.

    :param db_name: Dataset name (key in ``DATASET_COLUMNS``).
    :param which_measurement_field: ``"effect"``, ``"pvalue"``, or ``"log10pval"``.
    :return: Column name to use in queries.

    :raises ValueError: If ``which_measurement_field`` is invalid.
    :raises KeyError: If ``db_name`` is not in ``DATASET_COLUMNS``.

    """
    if which_measurement_field not in ("effect", "pvalue", "log10pval"):
        raise ValueError(f"Invalid measurement field: {which_measurement_field}")
    try:
        effect_col, pvalue_col, log10p_col, neglog10p_col = DATASET_COLUMNS[db_name]
    except KeyError as exc:
        raise KeyError(f"Unknown dataset name: {db_name}") from exc
    if which_measurement_field == "log10pval":
        return neglog10p_col or log10p_col or pvalue_col or effect_col
    if which_measurement_field == "pvalue":
        return pvalue_col or effect_col
    return effect_col


def get_log10p_source(
    db_name: str,
) -> Literal["neglog10p", "log10p", "pval", "none"]:
    """
    Return which source column provides the -log10(pval) value for a dataset.

    The caller uses this to determine what Python-side transform is needed after
    the query returns:

    - ``"neglog10p"``: column is already ``-log10(pval)`` — apply upper cap only.
    - ``"log10p"``: negate the column, then apply upper cap.
    - ``"pval"``: apply ``-log10(clip(lower=LOG10P_FLOOR))``.
    - ``"none"``: no p-value variant exists; -log10(pval) is unavailable.

    :param db_name: Dataset name (key in ``DATASET_COLUMNS``).
    :return: Source indicator string.

    :raises KeyError: If ``db_name`` is not in ``DATASET_COLUMNS``.

    """
    try:
        _, pvalue_col, log10p_col, neglog10p_col = DATASET_COLUMNS[db_name]
    except KeyError as exc:
        raise KeyError(f"Unknown dataset name: {db_name}") from exc
    if neglog10p_col:
        return "neglog10p"
    if log10p_col:
        return "log10p"
    if pvalue_col:
        return "pval"
    return "none"


def perturbation_data_query(
    db_name: str,
    col: str,
    filters: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    """
    Build a SELECT query and parameter dict for a perturbation dataset.

    Column names and table name are interpolated directly (not parameterizable
    via DuckDB). Filter values are parameterized with ``$name`` syntax.

    :param db_name: Name of the dataset table to query.
    :param col: Data column to select (in addition to the ID columns).
    :param filters: Optional filter spec; each key is a column name, each value is
        a dict with keys ``type`` (``"categorical"``, ``"numeric"``, or ``"bool"``)
        and ``value``.
    :return: Tuple of (sql_string, params_dict) ready to pass to ``vdb.query()``.

    """
    params: dict[str, Any] = {}
    where_clause = _meta_sample_where(db_name, filters, params) if filters else ""
    sql = (
        f"SELECT regulator_locus_tag, target_locus_tag, target_symbol, sample_id, {col} "
        f"FROM {db_name}{where_clause}"
    )
    return sql, params


def _build_where(
    filters: dict[str, Any],
    params: dict[str, Any],
    prefix: str = "",
) -> str:
    """
    Build a WHERE clause string and populate ``params`` in-place.

    :param filters: Filter spec dict (column -> {type, value}).
    :param params: Dict to populate with parameterized values.
    :param prefix: Namespace prefix to avoid collisions across datasets.
    :return: WHERE clause string (empty string if no filters).

    """
    if not filters:
        return ""
    clauses: list[str] = []
    for field, spec in filters.items():
        kind = spec["type"]
        val = spec["value"]
        p = (f"{prefix}_{field}" if prefix else field).replace(" ", "_")
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
    return f" WHERE {' AND '.join(clauses)}" if clauses else ""


def _meta_sample_where(
    db_name: str,
    filters: dict[str, Any] | None,
    params: dict[str, Any],
    prefix: str = "",
) -> str:
    """
    Build a ``WHERE sample_id IN (meta subquery)`` clause and populate ``params``.

    Dataset filters target metadata columns. Rather than predicating the wide
    data view (``{db_name}``) directly, the filter resolves to a ``sample_id``
    set against the small ``{db_name}_meta`` view, so the data-view scan only
    needs the projected columns and a ``sample_id`` membership test.

    :param db_name: Data view name; its meta view is ``{db_name}_meta``.
    :param filters: Filter spec dict (column -> {type, value}), or ``None``.
    :param params: Dict to populate with parameterized values.
    :param prefix: Namespace prefix to avoid collisions across datasets.
    :return: WHERE clause string (empty string if no filters).

    """
    if not filters:
        return ""
    inner = _build_where(filters, params, prefix)
    if not inner:
        return ""
    return f" WHERE sample_id IN (SELECT sample_id FROM {db_name}_meta{inner})"


def corr_pair_sql(
    vdb: VirtualDB,
    db_a: str,
    col_a: str,
    filters_a: dict[str, Any] | None,
    db_b: str,
    col_b: str,
    filters_b: dict[str, Any] | None,
    method: str,
    prefix: str = "",
    sql_only: bool = False,
) -> pd.DataFrame | tuple[str, dict[str, Any]]:
    """
    Compute per-regulator correlation between two perturbation datasets.

    Delegates to :func:`~tfbpshiny.modules.binding.queries._corr_pair_sql_impl`
    using :func:`perturbation_data_query`.
    See that function for full parameter and return documentation.

    """
    return _corr_pair_sql_impl(
        vdb,
        perturbation_data_query,
        db_a,
        col_a,
        filters_a,
        db_b,
        col_b,
        filters_b,
        method,
        prefix,
        sql_only,
    )


def corr_all_pairs_sql(
    vdb: VirtualDB,
    pairs: list[tuple[str, str]],
    col_map: dict[str, str],
    filters: dict[str, Any],
    method: str,
) -> pd.DataFrame:
    """
    Compute per-regulator correlations for all dataset pairs in a single query.

    Builds one UNION ALL query covering every pair and executes it as a single
    ``vdb.query()`` call, eliminating per-pair round-trip overhead.

    :param vdb: VirtualDB instance.
    :param pairs: List of ``(db_a, db_b)`` tuples.
    :param col_map: Mapping of ``db_name`` to the measurement column to use.
    :param filters: Active filter dict keyed by dataset name.
    :param method: ``"pearson"`` or ``"spearman"``.
    :return: DataFrame with columns ``db_a``, ``db_a_id``, ``db_b``, ``db_b_id``,
        ``regulator_locus_tag``, ``correlation``, and ``pair_key`` (``"{db_a}__{db_b}"``).

    """
    empty_cols = [
        "db_a",
        "db_a_id",
        "db_b",
        "db_b_id",
        "regulator_locus_tag",
        "correlation",
        "pair_key",
    ]
    if not pairs:
        return pd.DataFrame(columns=empty_cols)

    # Execute one pair at a time (not a single UNION ALL across pairs) so each
    # pair's join intermediates are released before the next runs, bounding peak
    # memory. The small per-(regulator, sample) correlation rows accumulate here.
    frames: list[pd.DataFrame] = []
    for db_a, db_b in pairs:
        pair_sql, pair_params = _corr_pair_sql_impl(
            vdb,
            perturbation_data_query,
            db_a,
            col_map[db_a],
            filters.get(db_a),
            db_b,
            col_map[db_b],
            filters.get(db_b),
            method,
            prefix="p",
            sql_only=True,
        )
        assert isinstance(pair_sql, str) and isinstance(pair_params, dict)
        df = vdb.query(pair_sql, **pair_params)
        if not df.empty:
            df["pair_key"] = f"{db_a}__{db_b}"
            frames.append(df)

    if not frames:
        return pd.DataFrame(columns=empty_cols)
    return pd.concat(frames, ignore_index=True)


def regulator_scatter_sql(
    db_a: str,
    col_a: str,
    filters_a: dict[str, Any] | None,
    db_b: str,
    col_b: str,
    filters_b: dict[str, Any] | None,
    method: str,
    regulator: str,
    idx: int,
) -> tuple[str, dict[str, Any]]:
    """
    Build a SELECT query returning per-target values for a single regulator, suitable
    for scatter plot rendering.

    For Pearson: returns raw column values as ``_val_a`` and ``_val_b``.
    For Spearman: returns ranks (effect ranked by ABS DESC, pvalue ranked ASC).

    :param db_a: First dataset name.
    :param col_a: Column to use from first dataset.
    :param filters_a: Optional filters for first dataset.
    :param db_b: Second dataset name.
    :param col_b: Column to use from second dataset.
    :param filters_b: Optional filters for second dataset.
    :param method: ``"pearson"`` or ``"spearman"``.
    :param regulator: Regulator locus tag to filter to.
    :param idx: Unique integer index to namespace parameters across multiple pairs.
    :return: Tuple of (sql_string, params_dict).

    """
    sql_a, params_a = perturbation_data_query(db_a, col_a, filters_a)
    sql_b, params_b = perturbation_data_query(db_b, col_b, filters_b)

    # Namespace filter params to avoid collisions when both datasets share
    # a filter field name (e.g. a common metadata column).
    prefix = f"rp{idx}"
    params_a = {f"{prefix}a_{k}": v for k, v in params_a.items()}
    params_b = {f"{prefix}b_{k}": v for k, v in params_b.items()}
    for old, new in [(k[len(f"{prefix}a_") :], k) for k in params_a]:
        sql_a = sql_a.replace(f"${old}", f"${new}")
    for old, new in [(k[len(f"{prefix}b_") :], k) for k in params_b]:
        sql_b = sql_b.replace(f"${old}", f"${new}")

    reg_key_a = f"{prefix}reg_a"
    reg_key_b = f"{prefix}reg_b"
    sql_a += (
        " AND " if "WHERE" in sql_a else " WHERE "
    ) + f"regulator_locus_tag = ${reg_key_a}"
    sql_b += (
        " AND " if "WHERE" in sql_b else " WHERE "
    ) + f"regulator_locus_tag = ${reg_key_b}"
    params_a[reg_key_a] = regulator
    params_b[reg_key_b] = regulator

    is_pvalue_a = "pval" in col_a.lower()
    is_pvalue_b = "pval" in col_b.lower()
    order_val_a = "val_a ASC" if is_pvalue_a else "ABS(val_a) DESC"
    order_val_b = "val_b ASC" if is_pvalue_b else "ABS(val_b) DESC"

    if method == "spearman":
        # Project qualified aliases first so ORDER BY is unambiguous even when
        # col_a == col_b (e.g. both datasets use the same column name).
        sql = f"""
            WITH a AS ({sql_a}), b AS ({sql_b}),
            joined AS (
              SELECT
                a.target_locus_tag,
                COALESCE(a.target_symbol, a.target_locus_tag) AS target_symbol,
                a.{col_a} AS val_a,
                b.{col_b} AS val_b
              FROM a JOIN b ON a.target_locus_tag = b.target_locus_tag
            )
            SELECT
              target_locus_tag,
              target_symbol,
              RANK() OVER (ORDER BY {order_val_a}) AS _val_a,
              RANK() OVER (ORDER BY {order_val_b}) AS _val_b
            FROM joined
        """
    else:
        sql = f"""
            WITH a AS ({sql_a}), b AS ({sql_b})
            SELECT
              a.target_locus_tag,
              COALESCE(a.target_symbol, a.target_locus_tag) AS target_symbol,
              a.{col_a} AS _val_a,
              b.{col_b} AS _val_b
            FROM a JOIN b ON a.target_locus_tag = b.target_locus_tag
        """

    return sql, {**params_a, **params_b}


__all__ = [
    "DATASET_COLUMNS",
    "LOG10P_FLOOR",
    "get_measurement_column",
    "get_log10p_source",
    "perturbation_data_query",
    "corr_pair_sql",
    "corr_all_pairs_sql",
    "regulator_scatter_sql",
]
