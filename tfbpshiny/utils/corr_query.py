"""Shared DuckDB-based correlation query helper for binding and perturbation workspaces."""

from __future__ import annotations

from typing import Any

import duckdb
import pandas as pd


def get_filtered_sample_ids(
    conn: duckdb.DuckDBPyConnection,
    db_name: str,
    filters: dict[str, Any] | None,
) -> list[str]:
    """
    Return CAST(sample_id AS VARCHAR) from {db_name}_meta matching filters.

    :param conn: Open read-only DuckDB connection.
    :param db_name: Dataset name whose ``{db_name}_meta`` table to query.
    :param filters: Filter spec dict (column -> {type, value}), or ``None``.
    :returns: List of sample ID strings.

    """
    where_clauses: list[str] = []
    params: list[Any] = []
    for field, spec in (filters or {}).items():
        kind = spec["type"]
        val = spec["value"]
        if kind == "categorical":
            phs = ", ".join(["?"] * len(val))
            where_clauses.append(f'CAST("{field}" AS VARCHAR) IN ({phs})')
            params.extend([str(v) for v in val])
        elif kind == "numeric":
            where_clauses.append(f'"{field}" BETWEEN ? AND ?')
            params.extend([val[0], val[1]])
        elif kind == "bool":
            where_clauses.append(f'"{field}" = ?')
            params.append(bool(val))
    where = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""
    sql = f"SELECT CAST(sample_id AS VARCHAR) AS sid FROM {db_name}_meta {where}"
    return conn.execute(sql, params).df()["sid"].tolist()


def fetch_corr_pairs(
    conn: duckdb.DuckDBPyConnection,
    pairs: list[tuple[str, str]],
    filters: dict[str, Any],
    method: str,
    comparison_type: str = "binding",
) -> dict[tuple[str, str], pd.DataFrame]:
    """
    Fetch pre-computed correlations from the correlations table for a list of pairs.

    :param conn: Read-only DuckDB connection to the materialized database.
    :param pairs: List of (db_a, db_b) dataset name pairs.
    :param filters: dataset_filters dict keyed by db_name.
    :param method: 'pearson' or 'spearman'.
    :param comparison_type: 'binding' or 'perturbation'.
    :returns: Dict mapping (db_a, db_b) to DataFrame with regulator_locus_tag, correlation.

    """
    result: dict[tuple[str, str], pd.DataFrame] = {}
    empty = pd.DataFrame(columns=["regulator_locus_tag", "correlation"])
    for db_a, db_b in pairs:
        try:
            row_a = (
                conn.execute(
                    "SELECT hf_repo, hf_config FROM dataset_registry WHERE db_name = ?",
                    [db_a],
                )
                .df()
                .iloc[0]
            )
            row_b = (
                conn.execute(
                    "SELECT hf_repo, hf_config FROM dataset_registry WHERE db_name = ?",
                    [db_b],
                )
                .df()
                .iloc[0]
            )
        except (IndexError, Exception):
            result[(db_a, db_b)] = empty.copy()
            continue
        prefix_a = f"{row_a['hf_repo']};{row_a['hf_config']};"
        prefix_b = f"{row_b['hf_repo']};{row_b['hf_config']};"

        ids_a = get_filtered_sample_ids(conn, db_a, filters.get(db_a))
        ids_b = get_filtered_sample_ids(conn, db_b, filters.get(db_b))

        if not ids_a or not ids_b:
            result[(db_a, db_b)] = empty.copy()
            continue

        phs_a = ", ".join(["?"] * len(ids_a))
        phs_b = ", ".join(["?"] * len(ids_b))
        sql = f"""
        SELECT regulator_locus_tag, correlation
        FROM correlations
        WHERE comparison_type = ?
          AND method = ?
          AND source_sample_a LIKE ?
          AND source_sample_b LIKE ?
          AND split_part(source_sample_a, ';', 3) IN ({phs_a})
          AND split_part(source_sample_b, ';', 3) IN ({phs_b})
        """
        params: list[Any] = (
            [comparison_type, method, prefix_a + "%", prefix_b + "%"] + ids_a + ids_b
        )
        try:
            df = conn.execute(sql, params).df()
        except Exception:
            df = empty.copy()
        result[(db_a, db_b)] = df
    return result


__all__ = ["get_filtered_sample_ids", "fetch_corr_pairs"]
