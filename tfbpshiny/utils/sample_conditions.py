"""
Helpers for attaching experimental-condition labels to individual samples.

The binding and perturbation distribution plots show one dot per sample pair. When
a regulator has multiple samples in a dataset the plot produces multiple dots for
that regulator, and the user needs a way to tell them apart in the tooltip. This
module builds ``{sample_id: condition_label}`` lookups from a dataset's ``_meta``
view so that the plot code can append human-readable condition text to each
selected-overlay dot's hover string.

``build_condition_label`` is a pure function and is the main unit-testable
surface. ``fetch_sample_condition_map`` is a thin VirtualDB wrapper that assembles
the SQL, runs it, and applies the label builder row-by-row.

"""

from __future__ import annotations

import re
from typing import Any

from labretriever import VirtualDB


def build_condition_label(values: list[Any]) -> str:
    """
    Combine one or more raw condition-column values into a single label.

    Non-empty, non-NaN string representations are joined with ``" / "``. A
    value of ``None``, a ``NaN`` float, or a string that is empty / whitespace
    / the literal ``"nan"`` (case-insensitive) is dropped. When every value
    drops out, an empty string is returned.

    :param values: One or more raw values from a single meta-table row.
    :returns: Display label, or ``""`` when nothing useful remains.

    """
    parts: list[str] = []
    for v in values:
        if v is None:
            continue
        # Catch float NaN without requiring a pandas/numpy import here.
        if isinstance(v, float) and v != v:
            continue
        s = str(v).strip()
        if not s or s.lower() == "nan":
            continue
        parts.append(s)
    return " / ".join(parts)


_SAFE_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def fetch_sample_condition_map(
    vdb: VirtualDB, db_name: str, cols: list[str]
) -> dict[str, str]:
    """
    Build a ``{sample_id: condition_label}`` map for one dataset.

    Queries ``{db_name}_meta`` for ``sample_id`` plus the requested condition
    columns and composes each row's label with :func:`build_condition_label`.
    Samples whose composed label is empty are omitted.

    Identifier safety: ``db_name`` must match a SQL-identifier pattern
    (``[A-Za-z_][A-Za-z0-9_]*``), otherwise a ``ValueError`` is raised before
    any SQL is built. Column names are double-quoted with embedded quotes
    doubled (DuckDB's identifier-escape), so columns containing spaces are
    fine but a column containing ``"`` cannot break out of the quoting.

    :param vdb: VirtualDB instance.
    :param db_name: Dataset name (the base name; ``_meta`` is appended).
        Must be a valid SQL identifier.
    :param cols: Condition column names, as taken from
        ``AppDatasets.condition_cols[db_name]``. May include spaces — each
        column is double-quoted in the generated SQL.
    :returns: Mapping from ``sample_id`` to its joined condition label. Empty
        when ``cols`` is empty.
    :raises ValueError: If ``db_name`` is not a safe identifier.

    """
    if not cols:
        return {}
    if not _SAFE_IDENT_RE.match(db_name):
        raise ValueError(f"db_name is not a safe SQL identifier: {db_name!r}")
    quoted = ", ".join(f'"{c.replace(chr(34), chr(34) * 2)}"' for c in cols)
    sql = f"SELECT sample_id, {quoted} FROM {db_name}_meta"
    df = vdb.query(sql)
    result: dict[str, str] = {}
    for _, row in df.iterrows():
        label = build_condition_label([row[c] for c in cols])
        if label:
            result[str(row["sample_id"])] = label
    return result


__all__ = ["build_condition_label", "fetch_sample_condition_map"]
