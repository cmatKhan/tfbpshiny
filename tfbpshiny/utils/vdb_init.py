"""One-time application initialization for VirtualDB and dataset metadata."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from logging import Logger

import pandas as pd
from labretriever import VirtualDB

from tfbpshiny.utils.profiler import profile_span

# Metadata fields to suppress from the filter UI, keyed by db_name.
# Use "*" for fields hidden across all datasets; use the db_name key for
# dataset-specific exclusions. The effective hidden set for a given dataset
# is the union of "*" and its own entry.
HIDDEN_FILTER_FIELDS: dict[str, set[str]] = {
    "*": {
        "regulator_locus_tag",
        "regulator_symbol",
        "Regulator locus tag",
        "Regulator symbol",
    },
    "callingcards": {"background_total_hops", "experiment_total_hops"},
    "harbison": {"condition"},
    "chec_m2025": {"condition", "mahendrawada_symbol"},
    "degron": {"env_condition", "timepoint"},
    "rossi": {"antibody", "growth_media"},
    "hackett": {"date", "mechanism", "restriction", "strain"},
    "hu_reimand": {"average_od_of_replicates", "heat_shock"},
    "hughes_overexpression": {"del_passed_qc", "sgd_description"},
    "hughes_knockout": {"oe_passed_qc", "sgd_description"},
}

# Datasets whose toggles are on by default. A superset of DEFAULT_DATASET_FILTERS
# — datasets with no preset conditions are listed here but not in the filter dict.
DEFAULT_ACTIVE_DATASETS: frozenset[str] = frozenset(
    {
        "harbison",
        "rossi",
        "chec_m2025",
        "hackett",
        "callingcards",
        "kemmeren",
        "degron",
    }
)

# Default filter state applied on first load. The structure is identical to the
# dict stored in the ``dataset_filters`` reactive value so it can be used as
# the initial value with no additional handling.
DEFAULT_DATASET_FILTERS: dict[str, dict] = {
    "harbison": {
        "condition": {"type": "categorical", "value": ["YPD"]},
    },
    "rossi": {
        "treatment": {"type": "categorical", "value": ["Normal"]},
    },
    "chec_m2025": {
        "Experimental condition": {"type": "categorical", "value": ["standard"]},
    },
    "hackett": {
        "time": {"type": "numeric", "value": [45.0, 45.0]},
    },
}

# NOTE: the following regulators have multiple samples with the same mechanism and
# restriction. For convenience for now, they are excluded since it is not possible
# to choose between them based on features in the dataset. They are essentially
# replicates and would need to be chosen by comparison to binding, for example
# 'GCN4', 'RDS2', 'SWI1', 'MAC1'
_HACKETT_ANALYSIS_SET_SQL = """
CREATE OR REPLACE TABLE hackett_analysis_set AS
WITH regulator_tiers AS (
    SELECT
        regulator_locus_tag,
        CASE
            WHEN BOOL_OR(mechanism = 'ZEV' AND restriction = 'P') THEN 1
            WHEN BOOL_OR(mechanism = 'GEV' AND restriction = 'P') THEN 2
            ELSE 3
        END AS tier
    FROM hackett_meta
    GROUP BY regulator_locus_tag
),
tier_filtered AS (
    SELECT
        h.sample_id,
        h.regulator_locus_tag,
        h.regulator_symbol,
        h.mechanism,
        h.restriction,
        h.time,
        h.date,
        h.strain,
        t.tier
    FROM hackett_meta h
    JOIN regulator_tiers t USING (regulator_locus_tag)
    WHERE
        (t.tier = 1 AND h.mechanism = 'ZEV' AND h.restriction = 'P')
        OR (t.tier = 2 AND h.mechanism = 'GEV' AND h.restriction = 'P')
        OR (t.tier = 3 AND h.mechanism = 'GEV' AND h.restriction = 'M')
)
SELECT DISTINCT
    sample_id,
    regulator_locus_tag,
    regulator_symbol,
    mechanism,
    restriction,
    time,
    date,
    strain
FROM tier_filtered
WHERE regulator_symbol NOT IN ('GCN4', 'RDS2', 'SWI1', 'MAC1')
"""


def _filter_hackett_views(vdb: VirtualDB) -> None:
    """
    Replace the ``hackett_meta`` and ``hackett`` DuckDB views with versions filtered to
    sample IDs present in ``hackett_analysis_set``.

    Extracts the original SELECT body from ``duckdb_views()`` and wraps it as an
    inline subquery, avoiding self-referential view replacement.

    :param vdb: The application VirtualDB instance.

    """
    conn = vdb._conn
    for view_name in ("hackett_meta", "hackett"):
        row = conn.execute(
            "SELECT sql FROM duckdb_views() WHERE view_name = ?", [view_name]
        ).fetchone()
        if row is None:
            continue
        full_sql: str = row[0]
        # full_sql is "CREATE VIEW view_name AS <select_body>;"
        # Split after the first " AS " to extract just the SELECT body.
        select_body = full_sql.split(" AS ", 1)[1].rstrip(";").strip()
        conn.execute(
            f"CREATE OR REPLACE VIEW {view_name} AS "
            f"SELECT * FROM ({select_body}) __base "
            f"WHERE sample_id IN (SELECT sample_id FROM hackett_analysis_set)"
        )


def ensure_hackett_analysis_set(
    vdb: VirtualDB, profile_logger: Logger | None = None
) -> None:
    """
    Build the ``hackett_analysis_set`` table and permanently filter the ``hackett_meta``
    and ``hackett`` views to include only those samples.

    Safe to call multiple times; uses ``CREATE OR REPLACE``.

    :param vdb: The application VirtualDB instance.
    :param profile_logger: Optional profiler logger; timing is skipped when ``None``.

    """
    _pl = profile_logger or logging.getLogger("profiler")
    with profile_span(_pl, "init.hackett_set"):
        vdb._conn.execute(_HACKETT_ANALYSIS_SET_SQL)
    _filter_hackett_views(vdb)


_REGULATOR_DISPLAY_NAME_TABLE = "regulator_display_names"

_BUILD_REGULATOR_DISPLAY_NAMES_SQL = """
CREATE OR REPLACE TABLE {table} AS
SELECT
    regulator_locus_tag,
    FIRST(regulator_symbol) AS regulator_symbol,
    CASE
        WHEN FIRST(regulator_symbol) IS NOT NULL
             AND FIRST(regulator_symbol) != ''
             AND FIRST(regulator_symbol) != FIRST(regulator_locus_tag)
        THEN FIRST(regulator_symbol) || ' (' || regulator_locus_tag || ')'
        ELSE regulator_locus_tag
    END AS display_name
FROM ({union_sql}) __all
GROUP BY regulator_locus_tag
ORDER BY regulator_locus_tag
"""


def _build_regulator_display_names(
    vdb: VirtualDB, profile_logger: Logger | None = None
) -> None:
    """
    Build the ``regulator_display_names`` DuckDB table from all dataset meta views.

    Queries each ``{db_name}_meta`` view for distinct ``(regulator_locus_tag,
    regulator_symbol)`` rows, unions them, and stores the result as a persistent
    in-memory table.  The ``display_name`` column is ``"SYMBOL (LOCUS_TAG)"`` when a
    non-empty symbol different from the tag is present; otherwise it equals the tag.

    :param vdb: The application VirtualDB instance.
    :param profile_logger: Optional profiler logger; timing is skipped when ``None``.

    """
    db_names = [
        db
        for db in vdb.get_datasets()
        if "regulator_locus_tag" in vdb.get_fields(f"{db}_meta")
    ]
    if not db_names:
        return
    union_sql = " UNION ALL ".join(
        f"SELECT DISTINCT regulator_locus_tag, regulator_symbol FROM {db}_meta"
        for db in db_names
    )
    sql = _BUILD_REGULATOR_DISPLAY_NAMES_SQL.format(
        table=_REGULATOR_DISPLAY_NAME_TABLE,
        union_sql=union_sql,
    )
    _pl = profile_logger or logging.getLogger("profiler")
    with profile_span(_pl, "init.regulator_table"):
        vdb._conn.execute(sql)


def get_regulator_display_name(
    vdb: VirtualDB,
    locus_tags: list[str] | None = None,
    profile_logger: Logger | None = None,
) -> pd.DataFrame:
    """
    Return a DataFrame of regulator display names from the pre-built lookup table.

    :param vdb: The application VirtualDB instance.
    :param locus_tags: Optional list of locus tags to restrict results. When
        ``None`` all regulators in the table are returned.
    :returns: DataFrame with columns ``regulator_locus_tag``, ``regulator_symbol``,
        and ``display_name``.
    :rtype: pandas.DataFrame

    """
    _pl = profile_logger or logging.getLogger("profiler")
    with profile_span(
        _pl,
        "vdb.execute",
        dataset=_REGULATOR_DISPLAY_NAME_TABLE,
        context="get_regulator_display_name",
    ):
        if locus_tags is None:
            return vdb._conn.execute(
                f"SELECT * FROM {_REGULATOR_DISPLAY_NAME_TABLE}"
            ).df()
        return vdb._conn.execute(
            f"SELECT * FROM {_REGULATOR_DISPLAY_NAME_TABLE} "
            f"WHERE regulator_locus_tag = ANY(?)",
            [locus_tags],
        ).df()


@dataclass
class AppDatasets:
    """
    App-level dataset metadata derived at startup.

    Holds the column classification that requires :data:`HIDDEN_FILTER_FIELDS`
    and cannot be produced by VirtualDB alone.

    :param condition_cols: Mapping from db_name to list of column names with
        role ``experimental_condition`` and non-None ``level_definitions``,
        excluding hidden fields.
    :param upstream_cols: Mapping from db_name to list of non-condition
        categorical columns that drive the cascade filter, excluding hidden
        fields, ``sample_id``, and identifier-role columns.

    """

    condition_cols: dict[str, list[str]]
    upstream_cols: dict[str, list[str]]


def initialize_data(
    virtualdb_config: str,
    hf_token: str | None = None,
    profile_logger: Logger | None = None,
) -> tuple[VirtualDB, AppDatasets]:
    """
    Construct the VirtualDB, run one-time setup, and compute app-level dataset metadata.

    :param virtualdb_config: Path to the VirtualDB YAML config file.
    :param hf_token: Optional HuggingFace token for private repo access.
    :param profile_logger: Optional profiler logger for timing instrumentation.
    :returns: Tuple of ``(vdb, app_datasets)``.
    :rtype: tuple[VirtualDB, AppDatasets]

    """
    vdb = VirtualDB(virtualdb_config, token=hf_token)
    ensure_hackett_analysis_set(vdb, profile_logger=profile_logger)
    _build_regulator_display_names(vdb, profile_logger=profile_logger)

    condition_cols: dict[str, list[str]] = {}
    upstream_cols: dict[str, list[str]] = {}
    hidden_global = HIDDEN_FILTER_FIELDS.get("*", set())

    for db_name in vdb.get_datasets():
        db_meta = vdb.get_column_metadata(db_name) or {}
        hidden = hidden_global | HIDDEN_FILTER_FIELDS.get(db_name, set())

        cond = [
            col
            for col, m in db_meta.items()
            if m.role == "experimental_condition"
            and m.level_definitions is not None
            and col not in hidden
        ]
        upstream = [
            col
            for col, m in db_meta.items()
            if col not in cond
            and col not in hidden
            and col != "sample_id"
            and m.role not in ("regulator_identifier", "target_identifier")
            and m.level_definitions is None
        ]
        if cond and upstream:
            condition_cols[db_name] = cond
            upstream_cols[db_name] = upstream

    return vdb, AppDatasets(condition_cols=condition_cols, upstream_cols=upstream_cols)


__all__ = [
    "HIDDEN_FILTER_FIELDS",
    "DEFAULT_ACTIVE_DATASETS",
    "DEFAULT_DATASET_FILTERS",
    "AppDatasets",
    "ensure_hackett_analysis_set",
    "get_regulator_display_name",
    "initialize_data",
]
