"""
Materialization coordinator.

Imports SQL generators from the submodules and executes them in dependency
order against both VirtualDB (data source) and the output DuckDB file (target).
Data is transferred as pandas DataFrames via ``vdb.query()`` → output
``conn.register()`` / ``conn.execute()``.
"""

from __future__ import annotations

import argparse
import itertools
import logging
import time
from typing import Any

import duckdb
from labretriever import VirtualDB

from tfbpshiny.materialize.comparison.correlations import (
    BINDING_DATASET_COLUMNS,
    PERTURBATION_DATASET_COLUMNS,
    correlation_pair_select_sql,
    correlations_schema_sql,
)
from tfbpshiny.materialize.comparison.dto import dto_select_sql
from tfbpshiny.materialize.comparison.topn import (
    BINDING_TOPN_CONFIGS,
    PERTURBATION_TOPN_DATASETS,
    topn_pair_select_sql,
    topn_schema_sql,
)
from tfbpshiny.materialize.coordinating.sql import (
    DATASET_HF_COORDS,
    binding_methods_sql,
    column_metadata_sql,
    comparative_registry_sql,
    dataset_registry_sql,
    promoter_sets_sql,
)
from tfbpshiny.materialize.metadata.sql import (
    meta_select_sql,
    regulator_display_names_select_sql,
)

logger = logging.getLogger("shiny")

_DEFAULT_REGULATORS_PER_CHUNK = 400


def _exec_static(conn: duckdb.DuckDBPyConnection, sql: str, label: str) -> None:
    """Execute a SQL block (possibly multi-statement) against the output connection."""
    t0 = time.monotonic()
    conn.execute(sql)
    logger.info("  %-40s  %.2fs", label, time.monotonic() - t0)


def _vdb_to_table(
    vdb: VirtualDB,
    output_conn: duckdb.DuckDBPyConnection,
    select_sql: str,
    params: dict[str, Any],
    target_table: str,
    label: str,
    mode: str = "create",
) -> int:
    """
    Execute a SELECT against VirtualDB and write the result to the output connection.

    :param vdb: VirtualDB instance (data source).
    :param output_conn: Output DuckDB connection (target).
    :param select_sql: SELECT SQL to execute against vdb.
    :param params: Named parameters for the SELECT (``$name`` syntax).
    :param target_table: Table name in the output database.
    :param label: Short label used in log messages.
    :param mode: ``'create'`` → ``CREATE TABLE … AS SELECT *``; ``'insert'`` →
        ``INSERT INTO … SELECT *``.
    :returns: Number of rows written.
    :rtype: int

    """
    t0 = time.monotonic()
    df = vdb.query(select_sql, **params)
    # DuckDB corr() can return NaN (a valid IEEE 754 float, not SQL NULL) for
    # zero-variance inputs.  The pandas→Arrow→DuckDB round-trip converts those
    # NaN floats to SQL NULL, which violates NOT NULL constraints.  Drop them
    # here so the SQL-level filter (`NOT isnan`) and this both agree.
    df = df.dropna(how="any")
    row_count = len(df)
    if row_count == 0:
        logger.debug("  %-40s  (0 rows, skipped)", label)
        return 0

    output_conn.register("_tmp_df", df)
    try:
        if mode == "create":
            output_conn.execute(
                f'CREATE TABLE "{target_table}" AS SELECT * FROM _tmp_df'
            )
        else:
            output_conn.execute(
                f'INSERT INTO "{target_table}" SELECT * FROM _tmp_df'
            )
    finally:
        output_conn.unregister("_tmp_df")

    logger.info(
        "  %-40s  %d rows  %.2fs",
        label,
        row_count,
        time.monotonic() - t0,
    )
    return row_count


def _regulators_for_binding(vdb: VirtualDB, binding_view: str) -> list[str]:
    """Return distinct regulator locus tags for a binding dataset (sorted)."""
    df = vdb.query(
        f"SELECT DISTINCT regulator_locus_tag FROM {binding_view}_meta"
    )
    return sorted(t for t in df["regulator_locus_tag"].dropna().tolist())


def materialize(
    output_path: str,
    vdb: VirtualDB,
    args: argparse.Namespace,
) -> None:
    """
    Run the full materialization pipeline, writing to ``output_path``.

    Execution order:
    1. Coordinating layer (registry tables, column metadata)
    2. Metadata layer (``{db_name}_meta`` tables, regulator display names)
    3. Comparison — HF-sourced (``dto``)
    4. Comparison — computed (``topn_results``, ``correlations``)

    :param output_path: Path to the output ``.duckdb`` file.
    :param vdb: VirtualDB instance (all dataset views registered).
    :param args: Parsed CLI args with fields ``methods``, ``top_n_values``,
        ``effect_thresholds``, ``pvalue_thresholds``, ``skip_topn``,
        ``skip_correlations``.

    """
    conn = duckdb.connect(output_path)
    t_total = time.monotonic()

    try:
        # ------------------------------------------------------------------
        # 1. Coordinating layer
        # ------------------------------------------------------------------
        logger.info("=== Phase 1: Coordinating layer ===")
        _exec_static(conn, promoter_sets_sql(), "promoter_sets")
        _exec_static(conn, binding_methods_sql(), "binding_methods")
        _exec_static(conn, dataset_registry_sql(), "dataset_registry")
        _exec_static(conn, comparative_registry_sql(), "comparative_dataset_registry")

        col_meta_sql = column_metadata_sql(vdb)
        _exec_static(conn, col_meta_sql, "dataset_column_metadata")

        # ------------------------------------------------------------------
        # 2. Metadata layer
        # ------------------------------------------------------------------
        logger.info("=== Phase 2: Metadata layer ===")
        datasets = vdb.get_datasets()
        meta_db_names: list[str] = []

        for db_name in datasets:
            meta_view = f"{db_name}_meta"
            row = vdb._conn.execute(
                "SELECT view_name FROM duckdb_views() WHERE view_name = ?",
                [meta_view],
            ).fetchone()
            if row is None:
                continue
            meta_db_names.append(db_name)
            _vdb_to_table(
                vdb,
                conn,
                meta_select_sql(db_name),
                {},
                meta_view,
                f"{meta_view}",
                mode="create",
            )

        reg_names = [
            db
            for db in meta_db_names
            if "regulator_locus_tag" in vdb.get_fields(f"{db}_meta")
        ]
        _vdb_to_table(
            vdb,
            conn,
            regulator_display_names_select_sql(reg_names),
            {},
            "regulator_display_names",
            "regulator_display_names",
            mode="create",
        )

        # ------------------------------------------------------------------
        # 3. Comparison — HF-sourced (DTO)
        # ------------------------------------------------------------------
        logger.info("=== Phase 3: HF-sourced comparison (DTO) ===")
        dto_view = vdb._conn.execute(
            "SELECT view_name FROM duckdb_views() WHERE view_name = 'dto'"
        ).fetchone()
        if dto_view is not None:
            _vdb_to_table(
                vdb, conn, dto_select_sql(), {}, "dto", "dto", mode="create"
            )
        else:
            logger.warning("  dto view not found in VirtualDB — skipping")

        # ------------------------------------------------------------------
        # 4. Comparison — computed
        # ------------------------------------------------------------------
        logger.info("=== Phase 4: Computed comparison tables ===")
        top_n_values: list[int] = args.top_n_values
        effect_thresholds: list[float] = args.effect_thresholds
        pvalue_thresholds: list[float] = args.pvalue_thresholds
        methods: list[str] = [m.strip() for m in args.methods.split(",")]
        chunk = _DEFAULT_REGULATORS_PER_CHUNK

        # ---- topn_results ----
        _exec_static(conn, topn_schema_sql(), "topn_results (schema)")

        if not args.skip_topn:
            binding_views = [
                db for db in datasets if db in BINDING_TOPN_CONFIGS
            ]
            perturbation_views = [
                db for db in datasets if db in PERTURBATION_TOPN_DATASETS
            ]

            for b_db, p_db in itertools.product(binding_views, perturbation_views):
                b_cfg = BINDING_TOPN_CONFIGS[b_db]
                b_hf_repo, b_hf_config = DATASET_HF_COORDS.get(b_db, ("", ""))
                p_hf_repo, p_hf_config = DATASET_HF_COORDS.get(p_db, ("", ""))

                regulators = _regulators_for_binding(vdb, b_db)
                batches: list[tuple[str, ...]] = [
                    tuple(regulators[i : i + chunk])
                    for i in range(0, len(regulators), chunk)
                ] or [()]

                for top_n, eff_thresh, pval_thresh in itertools.product(
                    top_n_values, effect_thresholds, pvalue_thresholds
                ):
                    pair_label = f"topn {b_db}×{p_db} n={top_n}"
                    pair_rows = 0
                    for batch_idx, batch in enumerate(batches):
                        sql, params = topn_pair_select_sql(
                            binding_view=b_db,
                            binding_hf_repo=b_hf_repo,
                            binding_hf_config=b_hf_config,
                            perturbation_view=p_db,
                            pert_hf_repo=p_hf_repo,
                            pert_hf_config=p_hf_config,
                            binding_sample_col=b_cfg["binding_sample_col"],
                            rank_col=b_cfg["rank_col"],
                            rank_asc=b_cfg["rank_asc"],
                            target_blacklist=b_cfg.get("target_blacklist", ()),
                            binding_dedup_cte=b_cfg.get("binding_dedup_cte", ""),
                            top_n=top_n,
                            effect_threshold=eff_thresh,
                            pvalue_threshold=pval_thresh,
                            regulator_subset=batch,
                            param_prefix=f"bp{batch_idx}",
                        )
                        batch_label = (
                            f"{pair_label} batch {batch_idx + 1}/{len(batches)}"
                        )
                        pair_rows += _vdb_to_table(
                            vdb,
                            conn,
                            sql,
                            params,
                            "topn_results",
                            batch_label,
                            mode="insert",
                        )
                    logger.info("  %-40s  total %d rows", pair_label, pair_rows)
        else:
            logger.info("  topn_results skipped (--skip-topn)")

        # ---- correlations ----
        _exec_static(conn, correlations_schema_sql(), "correlations (schema)")

        if not args.skip_correlations:
            # Binding × binding pairs (lexicographic order, each pair once)
            binding_corr_views = [
                db for db in datasets if db in BINDING_DATASET_COLUMNS
            ]
            for view_a, view_b in itertools.combinations(
                sorted(binding_corr_views), 2
            ):
                col_a = BINDING_DATASET_COLUMNS[view_a][0]
                col_b = BINDING_DATASET_COLUMNS[view_b][0]
                hf_a = DATASET_HF_COORDS.get(view_a, ("", ""))
                hf_b = DATASET_HF_COORDS.get(view_b, ("", ""))
                for method in methods:
                    sql, params = correlation_pair_select_sql(
                        view_a=view_a,
                        hf_repo_a=hf_a[0],
                        hf_config_a=hf_a[1],
                        col_a=col_a,
                        view_b=view_b,
                        hf_repo_b=hf_b[0],
                        hf_config_b=hf_b[1],
                        col_b=col_b,
                        method=method,
                        comparison_type="binding",
                    )
                    _vdb_to_table(
                        vdb,
                        conn,
                        sql,
                        params,
                        "correlations",
                        f"corr(binding) {view_a}×{view_b} [{method}]",
                        mode="insert",
                    )

            # Perturbation × perturbation pairs
            pert_corr_views = [
                db for db in datasets if db in PERTURBATION_DATASET_COLUMNS
            ]
            for view_a, view_b in itertools.combinations(
                sorted(pert_corr_views), 2
            ):
                col_a = PERTURBATION_DATASET_COLUMNS[view_a][0]
                col_b = PERTURBATION_DATASET_COLUMNS[view_b][0]
                hf_a = DATASET_HF_COORDS.get(view_a, ("", ""))
                hf_b = DATASET_HF_COORDS.get(view_b, ("", ""))
                for method in methods:
                    sql, params = correlation_pair_select_sql(
                        view_a=view_a,
                        hf_repo_a=hf_a[0],
                        hf_config_a=hf_a[1],
                        col_a=col_a,
                        view_b=view_b,
                        hf_repo_b=hf_b[0],
                        hf_config_b=hf_b[1],
                        col_b=col_b,
                        method=method,
                        comparison_type="perturbation",
                    )
                    _vdb_to_table(
                        vdb,
                        conn,
                        sql,
                        params,
                        "correlations",
                        f"corr(perturbation) {view_a}×{view_b} [{method}]",
                        mode="insert",
                    )
        else:
            logger.info("  correlations skipped (--skip-correlations)")

        logger.info(
            "Materialization complete in %.1fs → %s",
            time.monotonic() - t_total,
            output_path,
        )

    finally:
        conn.close()
