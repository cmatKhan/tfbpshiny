"""
Projected in-memory materialization of the data views used by the analyses.

VirtualDB registers each dataset as a lazy ``read_parquet()``-backed DuckDB view, so
every ``vdb.query()`` re-reads parquet from disk. On the deployment's slow disk this
dominates the Comparison "Execute Analysis" latency (the per-pair query scans the
multi-million-row binding/perturbation views several times each). Copying those views
into in-memory DuckDB tables once, at startup, makes the scans hit RAM instead — ~43%
faster locally and dramatically faster on slow disk.

To keep the footprint affordable, the data views are materialized **projected** to
only the columns the analyses reference (identity columns plus the per-source rank /
effect / pvalue / responsive measurement columns); the wide free-text/genomic metadata
columns are dropped. The small ``_meta`` views are materialized in full because dataset
filtering resolves a ``sample_id`` set against them.

Scope is the datasets the Comparison tab uses as binding or perturbation sources. The
projected column set unions every analysis module's measurement columns, so the same
materialized view also serves the Binding and Perturbation correlation/scatter tabs
(which read the same views) without dropping columns they need.

Materialization is on by default; disable it with ``TFBPSHINY_MATERIALIZE=0``.

"""

from __future__ import annotations

import logging
import os
import time

from labretriever import VirtualDB

import tfbpshiny.modules.binding.queries as binding_queries
import tfbpshiny.modules.perturbation.queries as perturbation_queries
from tfbpshiny.modules.comparison.queries import BINDING_CONFIGS

logger = logging.getLogger("shiny")

# Identity columns every analysis query selects, regardless of source.
_IDENTITY_COLUMNS: tuple[str, ...] = (
    "sample_id",
    "regulator_locus_tag",
    "regulator_symbol",
    "target_locus_tag",
    "target_symbol",
)

# Fallback responsive column used by the comparison responsive expression when a
# perturbation dataset has no effect/pvalue columns configured.
_RESPONSIVE_FALLBACK_COLUMN: str = "responsive"


def _is_comparison_data_view(db_name: str) -> bool:
    """
    Return whether a dataset is a binding or perturbation source for the analyses.

    :param db_name: Dataset name.
    :returns: True if the dataset participates as a binding or perturbation source.
    :rtype: bool

    """
    return (
        db_name in BINDING_CONFIGS
        or db_name in binding_queries.DATASET_COLUMNS
        or db_name in perturbation_queries.DATASET_COLUMNS
    )


def _view_columns(vdb: VirtualDB, view: str) -> set[str]:
    """Return the set of column names for a registered view."""
    return set(vdb._conn.execute(f"DESCRIBE {view}").fetchdf()["column_name"])


def _projected_columns(db_name: str, available: set[str]) -> set[str]:
    """
    Compute the column set to materialize for a data view.

    The set unions the identity columns, the comparison rank column, and every
    measurement column referenced by the comparison, binding-correlation, and
    perturbation-correlation paths, so one materialized view serves all of them.
    Columns absent from the view are dropped. No metadata/filter columns are
    included; filtering resolves against the (separately materialized) ``_meta``
    view.

    :param db_name: Dataset name.
    :param available: Columns actually present in the view.
    :returns: Column names to project, restricted to those present in the view.
    :rtype: set[str]

    """
    cols: set[str] = set(_IDENTITY_COLUMNS)

    binding_cfg = BINDING_CONFIGS.get(db_name)
    if binding_cfg is not None:
        cols.add(binding_cfg.get("rank_col", ""))

    # Measurement columns from the binding and perturbation correlation configs
    # (each is a 4-tuple of effect/pvalue/log10/neglog10 column names).
    for column_map in (
        binding_queries.DATASET_COLUMNS,
        perturbation_queries.DATASET_COLUMNS,
    ):
        cols.update(column_map.get(db_name, ()))

    if db_name in perturbation_queries.DATASET_COLUMNS:
        cols.add(_RESPONSIVE_FALLBACK_COLUMN)

    # Harbison's dedup CTE aggregates MIN(pvalue).
    if db_name == "harbison":
        cols.add("pvalue")

    return {c for c in cols if c and c in available}


def _materialize(vdb: VirtualDB, view: str, columns: set[str] | None) -> None:
    """
    Copy a view into an in-memory ``_mat_{view}`` table and repoint the view.

    :param vdb: VirtualDB instance.
    :param view: View name to materialize.
    :param columns: Column subset to project, or ``None`` to copy all columns.

    """
    select_list = (
        ", ".join(f'"{c}"' for c in sorted(columns)) if columns is not None else "*"
    )
    table = f"_mat_{view}"
    vdb._conn.execute(
        f"CREATE OR REPLACE TABLE {table} AS SELECT {select_list} FROM {view}"
    )
    vdb._conn.execute(f"CREATE OR REPLACE VIEW {view} AS SELECT * FROM {table}")


def materialize_comparison_views(vdb: VirtualDB) -> None:
    """
    Materialize the comparison data views (projected) and their meta views into RAM.

    Builds, for each binding/perturbation source dataset, an in-memory ``_mat_``
    table projected to the analysis columns and repoints the view at it; the
    matching ``_meta`` view is materialized in full. Disabled by setting
    ``TFBPSHINY_MATERIALIZE=0``. Safe to call once at startup.

    :param vdb: VirtualDB instance with all views registered.

    """
    if os.getenv("TFBPSHINY_MATERIALIZE", "1") == "0":
        logger.debug(
            "materialize_comparison_views: disabled via TFBPSHINY_MATERIALIZE=0"
        )
        return

    t0 = time.monotonic()
    datasets = vdb.get_datasets()
    data_views = [db for db in datasets if _is_comparison_data_view(db)]

    for view in data_views:
        available = _view_columns(vdb, view)
        columns = _projected_columns(view, available)
        _materialize(vdb, view, columns)
        meta_view = f"{view}_meta"
        row = vdb._conn.execute(
            "SELECT view_name FROM duckdb_views() WHERE view_name = ?",
            [meta_view],
        ).fetchone()
        if row is not None:
            _materialize(vdb, meta_view, None)
        logger.debug(
            "materialize_comparison_views: %s projected to %d/%d columns",
            view,
            len(columns),
            len(available),
        )

    logger.debug(
        "materialize_comparison_views: completed in %.3fs (%d data views)",
        time.monotonic() - t0,
        len(data_views),
    )
