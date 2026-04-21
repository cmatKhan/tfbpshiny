"""Unit tests for ``tfbpshiny.utils.sample_conditions``."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from tfbpshiny.utils.sample_conditions import (
    build_condition_label,
    fetch_sample_condition_map,
)

# ---------- build_condition_label ----------


@pytest.mark.parametrize(
    "values,expected",
    [
        (["YPD"], "YPD"),
        (["ZEV", "P", "45"], "ZEV / P / 45"),
        (["ZEV", None, "45"], "ZEV / 45"),
        ([None, None], ""),
        ([], ""),
        (["  spaced  "], "spaced"),
        (["", "  ", "YPD"], "YPD"),
        (["nan", "NaN", "YPD"], "YPD"),
        ([float("nan"), "YPD"], "YPD"),
        ([1, 2.5, "x"], "1 / 2.5 / x"),
    ],
)
def test_build_condition_label(values: list[Any], expected: str) -> None:
    assert build_condition_label(values) == expected


# ---------- fetch_sample_condition_map ----------


class _StubVdb:
    """Minimal stub capturing the last-issued SQL and returning a canned DataFrame."""

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df
        self.last_sql: str | None = None

    def query(self, sql: str, **params: Any) -> pd.DataFrame:
        self.last_sql = sql
        return self._df


def test_fetch_sample_condition_map_empty_cols_returns_empty() -> None:
    vdb = _StubVdb(pd.DataFrame())
    assert fetch_sample_condition_map(vdb, "anything", []) == {}
    # No query should have been issued.
    assert vdb.last_sql is None


def test_fetch_sample_condition_map_single_column() -> None:
    df = pd.DataFrame(
        {
            "sample_id": ["s1", "s2", "s3"],
            "Experimental condition": ["YPD", "HEAT", None],
        }
    )
    vdb = _StubVdb(df)
    result = fetch_sample_condition_map(vdb, "harbison", ["Experimental condition"])

    assert result == {"s1": "YPD", "s2": "HEAT"}
    # s3 had an all-NULL label and must not be present.
    assert "s3" not in result
    # Column with a space is double-quoted in the SQL.
    assert vdb.last_sql is not None
    assert '"Experimental condition"' in vdb.last_sql
    assert "FROM harbison_meta" in vdb.last_sql


def test_fetch_sample_condition_map_multi_column_joined() -> None:
    df = pd.DataFrame(
        {
            "sample_id": ["s1", "s2"],
            "mechanism": ["ZEV", "GEV"],
            "restriction": ["P", None],
        }
    )
    vdb = _StubVdb(df)
    result = fetch_sample_condition_map(vdb, "hackett", ["mechanism", "restriction"])

    assert result == {"s1": "ZEV / P", "s2": "GEV"}


def test_fetch_sample_condition_map_non_string_sample_id_coerced() -> None:
    # sample_id comes back from DuckDB as int or bytes in some schemas.
    df = pd.DataFrame({"sample_id": [1, 2], "cond": ["A", "B"]})
    vdb = _StubVdb(df)
    result = fetch_sample_condition_map(vdb, "ds", ["cond"])
    assert result == {"1": "A", "2": "B"}


@pytest.mark.parametrize(
    "bad_name",
    [
        "foo; DROP TABLE users",
        "foo-bar",
        "foo bar",
        "1foo",
        "",
        'foo"; SELECT',
    ],
)
def test_fetch_sample_condition_map_rejects_unsafe_db_name(bad_name: str) -> None:
    vdb = _StubVdb(pd.DataFrame({"sample_id": ["s1"], "cond": ["A"]}))
    with pytest.raises(ValueError, match="safe SQL identifier"):
        fetch_sample_condition_map(vdb, bad_name, ["cond"])


def test_fetch_sample_condition_map_escapes_embedded_quotes_in_cols() -> None:
    # Column name with an embedded double-quote must be escaped as "" per
    # DuckDB identifier rules; otherwise the SQL would break out of quoting.
    df = pd.DataFrame({"sample_id": ["s1"], 'weird"name': ["A"]})
    vdb = _StubVdb(df)
    result = fetch_sample_condition_map(vdb, "ds", ['weird"name'])
    assert result == {"s1": "A"}
    assert vdb.last_sql is not None
    # The emitted identifier is "weird""name" — one escaped embedded quote.
    assert '"weird""name"' in vdb.last_sql
