"""Unit tests for select_datasets pure helper functions."""

from tfbpshiny.modules.select_datasets.queries import (
    _build_where,
    metadata_query,
    regulator_breakdown_query,
    regulator_display_labels_query,
    regulator_locus_tags_query,
    sample_count_query,
)

# --- _build_where ---


def test_build_where_no_filters():
    params: dict = {}
    assert _build_where(None, params) == ""
    assert params == {}


def test_build_where_categorical():
    params: dict = {}
    where = _build_where(
        {"strain": {"type": "categorical", "value": ["BY4741"]}}, params
    )
    assert '"strain" IN' in where
    assert "$cat_strain_0" in where
    assert params["cat_strain_0"] == "BY4741"


def test_build_where_numeric():
    params: dict = {}
    where = _build_where({"time": {"type": "numeric", "value": [0.0, 30.0]}}, params)
    assert "BETWEEN" in where
    assert params["num_time_lo"] == 0.0
    assert params["num_time_hi"] == 30.0


def test_build_where_bool():
    params: dict = {}
    where = _build_where({"is_wt": {"type": "bool", "value": True}}, params)
    assert "is_wt" in where
    assert params["bool_is_wt"] is True


# --- query builders ---


def test_metadata_query_no_filters():
    sql, params = metadata_query("harbison")
    assert sql == "SELECT * FROM harbison_meta"
    assert params == {}


def test_metadata_query_with_filter():
    sql, params = metadata_query(
        "harbison", {"strain": {"type": "categorical", "value": ["BY4741"]}}
    )
    assert "WHERE" in sql
    assert params["cat_strain_0"] == "BY4741"


def test_sample_count_query():
    sql, params = sample_count_query("harbison")
    assert "COUNT(sample_id)" in sql
    assert params == {}


def test_sample_count_query_with_regulators():
    sql, params = sample_count_query("harbison", restrict_to_regulators=["YAL001C"])
    assert "regulator_locus_tag IN" in sql
    assert "YAL001C" in params.values()


def test_regulator_locus_tags_query():
    sql, params = regulator_locus_tags_query("harbison")
    assert "DISTINCT regulator_locus_tag" in sql
    assert params == {}


def test_regulator_display_labels_query():
    sql, params = regulator_display_labels_query("harbison")
    assert "regulator_locus_tag" in sql
    assert "regulator_symbol" in sql
    assert "harbison_meta" in sql
    assert params == {}


# --- regulator_breakdown_query ---


def test_regulator_breakdown_query_no_filters_no_cols():
    sql, params = regulator_breakdown_query("harbison", [])
    assert "n_multi" in sql
    assert "harbison_meta" in sql
    assert "HAVING COUNT(*) > 1" in sql
    assert params == {}


def test_regulator_breakdown_query_candidate_cols_in_select():
    sql, params = regulator_breakdown_query(
        "harbison", ["Carbon source", "Temperature"]
    )
    assert 'COUNT(DISTINCT "Carbon source")' in sql
    assert 'COUNT(DISTINCT "Temperature")' in sql
    assert params == {}


def test_regulator_breakdown_query_with_filters():
    sql, params = regulator_breakdown_query(
        "harbison",
        ["Carbon source"],
        {"strain": {"type": "categorical", "value": ["BY4741"]}},
    )
    assert params["cat_strain_0"] == "BY4741"
    assert 'COUNT(DISTINCT "Carbon source")' in sql
    assert "HAVING COUNT(*) > 1" in sql
    # Two-CTE pattern: multi and per_reg both query harbison_meta.
    assert sql.count("FROM harbison_meta") >= 1


def test_regulator_breakdown_query_no_filters_uses_having():
    sql, params = regulator_breakdown_query("harbison", ["Carbon source"])
    # No filters — multi-sample filter is HAVING COUNT(*) > 1.
    assert "HAVING COUNT(*) > 1" in sql
    # The two-CTE pattern uses multi + per_reg, each querying harbison_meta.
    assert "harbison_meta" in sql
