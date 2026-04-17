"""One-time application initialization for VirtualDB and dataset metadata."""

from __future__ import annotations

from dataclasses import dataclass

from labretriever import VirtualDB

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


def ensure_hackett_analysis_set(vdb: VirtualDB) -> None:
    """
    Build the ``hackett_analysis_set`` table and permanently filter the ``hackett_meta``
    and ``hackett`` views to include only those samples.

    Safe to call multiple times; uses ``CREATE OR REPLACE``.

    :param vdb: The application VirtualDB instance.

    """
    vdb._conn.execute(_HACKETT_ANALYSIS_SET_SQL)
    _filter_hackett_views(vdb)


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
) -> tuple[VirtualDB, AppDatasets]:
    """
    Construct the VirtualDB, run one-time setup, and compute app-level dataset metadata.

    :param virtualdb_config: Path to the VirtualDB YAML config file.
    :param hf_token: Optional HuggingFace token for private repo access.
    :returns: Tuple of ``(vdb, app_datasets)``.
    :rtype: tuple[VirtualDB, AppDatasets]

    """
    vdb = VirtualDB(virtualdb_config, token=hf_token)
    ensure_hackett_analysis_set(vdb)

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
    "AppDatasets",
    "ensure_hackett_analysis_set",
    "initialize_data",
]
