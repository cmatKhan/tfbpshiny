"""Unit tests for the dataset export helpers."""

from __future__ import annotations

import tarfile
from unittest.mock import MagicMock

import duckdb
import pandas as pd
import pytest

from tfbpshiny.modules.select_datasets.export import (
    ExportDataset,
    _query_to_csv_bytes,
    _safe_dir_name,
    build_export_tarball,
    build_readme,
    get_dataset_description,
)

# --- _safe_dir_name ---


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("2026 Calling Cards", "2026_Calling_Cards"),
        ("simple", "simple"),
        ("has-hyphen", "has-hyphen"),
        ("has.dot", "has.dot"),
        ("../evil", "evil"),
        ("../../etc/passwd", "etc_passwd"),
        ("/absolute/path", "absolute_path"),
        ("name\x00null", "name_null"),
        ("a/b", "a_b"),
        ("   spaces   ", "spaces"),
        ("...", "dataset"),
        ("___", "dataset"),
        ("", "dataset"),
    ],
)
def test_safe_dir_name(raw: str, expected: str) -> None:
    assert _safe_dir_name(raw) == expected


# --- build_readme ---


def test_build_readme_includes_display_name():
    result = build_readme("2026 Calling Cards", "A binding dataset.")
    assert "# 2026 Calling Cards" in result


def test_build_readme_includes_description():
    result = build_readme("Test", "Some description text.")
    assert "Some description text." in result


def test_build_readme_includes_file_explanations():
    result = build_readme("Test", "desc")
    assert "metadata.csv" in result
    assert "annotated_features.csv" in result
    assert "Sample-level metadata" in result
    assert "Full genomic data" in result


# --- get_dataset_description ---


def test_get_dataset_description_delegates_to_vdb():
    """get_dataset_description passes through the return value of
    vdb.get_dataset_description."""
    vdb = MagicMock()
    vdb.get_dataset_description.return_value = "Harbison ChIP-chip data."
    assert get_dataset_description(vdb, "harbison") == "Harbison ChIP-chip data."
    vdb.get_dataset_description.assert_called_once_with("harbison")


def test_get_dataset_description_returns_none_when_vdb_returns_none():
    vdb = MagicMock()
    vdb.get_dataset_description.return_value = None
    assert get_dataset_description(vdb, "harbison") is None


# --- DuckDB-backed test fixtures ---


@pytest.fixture
def duckdb_vdb():
    """
    Minimal VirtualDB stub backed by a real in-memory DuckDB connection.

    Mirrors export.py's use of vdb._conn — update both if VirtualDB changes.

    """
    conn = duckdb.connect(":memory:")
    conn.execute(
        """
        CREATE TABLE test_meta AS
        SELECT 's1' AS sample_id, 'BY4741' AS strain
        UNION ALL
        SELECT 's2', 'BY4741'
        """
    )
    conn.execute(
        """
        CREATE TABLE test_data AS
        SELECT 's1' AS sample_id, 'YAL001C' AS gene, 1.0 AS value
        UNION ALL
        SELECT 's1', 'YAL002W', 2.0
        """
    )
    vdb = MagicMock()
    vdb._conn = conn
    yield vdb
    conn.close()


def _open_tarball(buf):
    """Open a BytesIO buffer as a tarball for reading."""
    buf.seek(0)
    return tarfile.open(fileobj=buf, mode="r:gz")


# --- _query_to_csv_bytes ---


def test_query_to_csv_bytes(duckdb_vdb):
    cursor = duckdb_vdb._conn.cursor()
    csv_bytes = _query_to_csv_bytes(cursor, "SELECT * FROM test_meta", {})
    text = csv_bytes.decode("utf-8")
    assert "sample_id" in text
    assert "BY4741" in text
    lines = text.strip().split("\n")
    assert len(lines) == 3  # header + 2 rows


def test_query_to_csv_bytes_uses_lf_line_endings(duckdb_vdb):
    """csv.writer must produce LF (not CRLF) for Unix tool compatibility."""
    cursor = duckdb_vdb._conn.cursor()
    csv_bytes = _query_to_csv_bytes(cursor, "SELECT * FROM test_meta", {})
    assert b"\r\n" not in csv_bytes
    assert b"\n" in csv_bytes


# --- build_export_tarball ---


def _make_export_dataset(
    display_name: str = "Test Dataset",
    description: str | None = None,
) -> ExportDataset:
    return ExportDataset(
        display_name=display_name,
        metadata_sql="SELECT * FROM test_meta",
        metadata_params={},
        data_sql="SELECT * FROM test_data",
        data_params={},
        description=description,
    )


def test_single_dataset_with_description(duckdb_vdb):
    ds = _make_export_dataset(
        display_name="2026 Calling Cards",
        description="A binding dataset.",
    )
    buf = build_export_tarball([ds], duckdb_vdb)

    with _open_tarball(buf) as tar:
        names = tar.getnames()
        assert "2026_Calling_Cards/metadata.csv" in names
        assert "2026_Calling_Cards/annotated_features.csv" in names
        assert "2026_Calling_Cards/README.md" in names


def test_single_dataset_without_description(duckdb_vdb):
    ds = _make_export_dataset(display_name="Test Dataset")
    buf = build_export_tarball([ds], duckdb_vdb)

    with _open_tarball(buf) as tar:
        names = tar.getnames()
        assert "Test_Dataset/metadata.csv" in names
        assert "Test_Dataset/annotated_features.csv" in names
        assert "Test_Dataset/README.md" not in names


def test_multiple_datasets(duckdb_vdb):
    ds1 = _make_export_dataset(display_name="Dataset A", description="First.")
    ds2 = _make_export_dataset(display_name="Dataset B")
    buf = build_export_tarball([ds1, ds2], duckdb_vdb)

    with _open_tarball(buf) as tar:
        names = tar.getnames()
        assert "Dataset_A/metadata.csv" in names
        assert "Dataset_B/metadata.csv" in names
        assert "Dataset_A/README.md" in names
        assert "Dataset_B/README.md" not in names


def test_csv_content_matches_query(duckdb_vdb):
    ds = _make_export_dataset(display_name="Check")
    buf = build_export_tarball([ds], duckdb_vdb)

    with _open_tarball(buf) as tar:
        meta_member = tar.extractfile("Check/metadata.csv")
        assert meta_member is not None
        recovered_meta = pd.read_csv(meta_member)
        assert list(recovered_meta.columns) == ["sample_id", "strain"]
        assert len(recovered_meta) == 2

        data_member = tar.extractfile("Check/annotated_features.csv")
        assert data_member is not None
        recovered_data = pd.read_csv(data_member)
        assert "gene" in recovered_data.columns
        assert len(recovered_data) == 2


def test_empty_dataset_list(duckdb_vdb):
    buf = build_export_tarball([], duckdb_vdb)
    with _open_tarball(buf) as tar:
        assert tar.getnames() == []


def test_tarball_no_path_traversal(duckdb_vdb):
    """Verify that adversarial display_names never produce traversal paths."""
    ds = _make_export_dataset(
        display_name="../../../etc/cron.d/backdoor",
    )
    buf = build_export_tarball([ds], duckdb_vdb)

    with _open_tarball(buf) as tar:
        for member in tar.getmembers():
            assert not member.name.startswith("/")
            assert ".." not in member.name.split("/")


def test_progress_callback_called(duckdb_vdb):
    """Verify the progress callback fires once per dataset."""
    ds1 = _make_export_dataset(display_name="A")
    ds2 = _make_export_dataset(display_name="B")
    names: list[str] = []
    build_export_tarball([ds1, ds2], duckdb_vdb, progress_callback=names.append)
    assert names == ["A", "B"]
