"""
SQL generators for the DTO (Directional Transcription Overlap) comparison table.

The DTO data is sourced verbatim from the HuggingFace Parquet stored in
``BrentLab/yeast_comparative_analysis;dto``.  The raw composite
``source_sample`` identifiers are preserved exactly as stored upstream.
"""

from __future__ import annotations


def dto_schema_sql() -> str:
    """
    Return the ``CREATE TABLE dto`` DDL (empty — no data).

    :returns: ``CREATE TABLE dto (…)`` SQL string.
    :rtype: str

    """
    return """
CREATE TABLE dto (
    binding_id              VARCHAR  NOT NULL,
    perturbation_id         VARCHAR  NOT NULL,
    dto_empirical_pvalue    DOUBLE,
    dto_fdr                 DOUBLE,
    binding_set_size        INTEGER,
    perturbation_set_size   INTEGER,
    pr_ranking_column       VARCHAR,
    PRIMARY KEY (binding_id, perturbation_id, pr_ranking_column)
);
"""


def dto_select_sql() -> str:
    """
    Return a SELECT that reads the raw DTO view from VirtualDB.

    The coordinator wraps this as::

        CREATE TABLE dto AS {dto_select_sql()}

    The raw ``dto`` view (not ``dto_expanded``) preserves the composite
    ``source_sample`` identifiers in the ``binding_id`` and ``perturbation_id``
    columns, matching the schema exactly.

    :returns: ``SELECT * FROM dto`` SQL string.
    :rtype: str

    """
    return "SELECT * FROM dto"
