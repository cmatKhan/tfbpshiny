"""
SQL generators for the metadata layer of the materialized DuckDB schema.

Functions return SELECT SQL strings suitable for wrapping in
``CREATE TABLE {name} AS {sql}`` by the coordinator.  Each can be called
independently in a Jupyter notebook to inspect what would be materialized.
"""

from __future__ import annotations


def meta_select_sql(db_name: str) -> str:
    """
    Return a SELECT statement that reads all rows from a dataset's meta view.

    The coordinator wraps this as::

        CREATE TABLE {db_name}_meta AS {meta_select_sql(db_name)}

    :param db_name: Dataset name (e.g. ``'callingcards'``).
    :returns: ``SELECT * FROM {db_name}_meta`` SQL string.
    :rtype: str

    """
    return f"SELECT * FROM {db_name}_meta"


def regulator_display_names_select_sql(db_names: list[str]) -> str:
    """
    Return a SELECT statement that builds the ``regulator_display_names`` table.

    Unions ``(regulator_locus_tag, regulator_symbol)`` rows from every
    ``{db_name}_meta`` view, then groups by locus tag to produce a single
    ``display_name`` per regulator.  The coordinator wraps this as::

        CREATE TABLE regulator_display_names AS {sql}

    :param db_names: List of dataset names whose ``_meta`` views have a
        ``regulator_locus_tag`` column.
    :returns: SELECT SQL that produces ``(regulator_locus_tag, regulator_symbol,
        display_name)`` rows.
    :rtype: str

    """
    if not db_names:
        return """
SELECT
    NULL::VARCHAR AS regulator_locus_tag,
    NULL::VARCHAR AS regulator_symbol,
    NULL::VARCHAR AS display_name
WHERE FALSE
"""
    union_parts = [
        f"SELECT DISTINCT regulator_locus_tag, regulator_symbol FROM {db}_meta"
        for db in db_names
    ]
    union_sql = " UNION ALL ".join(union_parts)
    return f"""
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
