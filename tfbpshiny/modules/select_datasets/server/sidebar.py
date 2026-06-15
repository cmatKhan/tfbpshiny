"""Sidebar server for the Select Datasets page."""

from __future__ import annotations

from collections.abc import Callable
from logging import Logger
from typing import Any

import pandas as pd
from labretriever import ColumnMeta, VirtualDB
from shiny import reactive, render, req, ui
from shiny.types import SilentException

from tfbpshiny.modules.select_datasets.queries import (
    metadata_query,
)
from tfbpshiny.modules.select_datasets.server.dataset_row import (
    dataset_row_server,
    dataset_row_ui,
)
from tfbpshiny.modules.select_datasets.ui import _slugify
from tfbpshiny.utils.perf import perf, reset_render_counts
from tfbpshiny.utils.vdb_init import (
    DEFAULT_ACTIVE_DATASETS,
    DEFAULT_DATASET_FILTERS,
    FIELD_TYPE_OVERRIDES,
    PRIMARY_DATASETS,
    AppDatasets,
)


def _build_experimental_condition_field_choices(
    df: pd.DataFrame,
    mask: pd.Series[bool],
    condition_cols: list[str],
    db_meta: dict[str, ColumnMeta],
) -> dict[str, dict[str, str]]:
    """
    Return condition column choices filtered by mask, sorted by descending count.

    :param df: Full metadata DataFrame for the dataset.
    :param mask: Boolean mask to apply before counting levels.
    :param condition_cols: Column names with role ``experimental_condition``.
    :param db_meta: Per-column metadata for the dataset.
    :returns: Dict mapping condition column name to ``{value: label}`` choices.

    """
    result: dict[str, dict[str, str]] = {}
    for cond_col in condition_cols:
        if cond_col not in df.columns:
            continue
        valid = (
            df.loc[mask, cond_col].dropna().astype(str).value_counts().index.tolist()
        )
        col_m = db_meta.get(cond_col)
        level_defs = col_m.level_definitions if col_m else {}
        result[cond_col] = {
            v: (f"{level_defs[v]} ({v})" if level_defs and level_defs.get(v) else v)
            for v in valid
        }
    return result


def select_datasets_sidebar_server(
    input: Any,
    output: Any,
    session: Any,
    vdb: VirtualDB,
    app_datasets: AppDatasets,
    logger: Logger,
    active_tab: reactive.Calc_[str] | None = None,
    pending_regulator_pair: reactive.Value[dict[str, Any] | None] | None = None,
    materialize_ready: Callable[[], bool] | None = None,
) -> tuple[
    reactive.Calc_[list[str]],
    reactive.Calc_[list[str]],
    reactive.Value[dict[str, Any]],
]:
    """
    Render dataset selection sidebar; return (active_binding_datasets,
    active_perturbation_datasets, dataset_filters).

    The sidebar has two sections: "Binding" and "Perturbation".
    Datasets are sourced from VirtualDB tags (data_type, display_name).

    """

    # dataset_dict: {db_name: {"data_type": "binding"|"perturbation",
    #                           "display_name": str, "assay": str, ...}}
    # Kept as a lookup map for display_name (passed to dataset_row_server
    # for the filter-modal title). The two
    # sorted lists below are derived views that provide rendering order and
    # the active-dataset calcs. They also carry description so it is
    # fetched once at startup rather than on every render.
    dataset_dict: dict[str, dict[str, str]] = {}
    for db_name in vdb.get_datasets():
        tags = vdb.get_tags(db_name)
        if (
            tags.get("data_type") in ["binding", "perturbation"]
            and db_name in PRIMARY_DATASETS
        ):
            dataset_dict[db_name] = tags

    # list of (db_name, display_name, description) tuples, sorted by year
    # (display_name always starts with the 4-digit year)
    binding_datasets: list[tuple[str, str, str]] = sorted(
        [
            (
                db_name,
                tags.get("display_name", db_name),
                vdb.get_dataset_description(db_name) or "",
            )
            for db_name, tags in dataset_dict.items()
            if tags.get("data_type") == "binding"
        ],
        key=lambda t: t[1],
    )
    perturbation_datasets: list[tuple[str, str, str]] = sorted(
        [
            (
                db_name,
                tags.get("display_name", db_name),
                vdb.get_dataset_description(db_name) or "",
            )
            for db_name, tags in dataset_dict.items()
            if tags.get("data_type") == "perturbation"
        ],
        key=lambda t: t[1],
    )
    # there are some common fields across datasets. In the dataset filters,
    # these common fields are displayed in their own section of the modal, and when
    # they are set on any dataset, they are applied to all datasets.
    common_fields = set(vdb.get_common_fields()) - {"sample_id"}

    # Per-column metadata from DataCards: descriptions, roles, level definitions.
    # {db_name: {col_name: ColumnMeta}}
    all_col_meta: dict[str, dict[str, ColumnMeta]] = {
        _db: (vdb.get_column_metadata(_db) or {}) for _db in dataset_dict
    }

    session.on_flush(lambda: reset_render_counts(session.id))

    _initial_toggle: dict[str, bool] = {
        db_name: db_name in DEFAULT_ACTIVE_DATASETS
        for db_name, _, _ in binding_datasets + perturbation_datasets
    }

    # Two-state model: committed (drives expensive calcs) and pending (staging).
    # Pending receives all user edits immediately; committed is updated only when
    # the user clicks Apply. The Apply button is shown whenever pending != committed.
    # {<db_name>: {<field_name>: {"type": "categorical" or "numeric" or "bool",
    #                              "value": list[str] | [lo, hi] | bool}}}
    dataset_filters: reactive.Value[dict[str, Any]] = reactive.value(
        DEFAULT_DATASET_FILTERS
    )
    _committed_toggle: reactive.Value[dict[str, bool]] = reactive.value(_initial_toggle)

    _pending_filters: reactive.Value[dict[str, Any]] = reactive.value(
        DEFAULT_DATASET_FILTERS
    )
    _pending_toggle: reactive.Value[dict[str, bool]] = reactive.value(_initial_toggle)

    # tracks which db_name's filter modal is currently open
    modal_open_for: reactive.Value[str | None] = reactive.value(None)
    # stores the DataFrame fetched when a filter modal is opened
    modal_df: reactive.Value[pd.DataFrame | None] = reactive.value(None)

    @reactive.calc
    def _has_pending_changes() -> bool:
        """
        True when the pending state differs from the committed state.

        :trigger: ``_pending_toggle``, ``_pending_filters``,
            ``_committed_toggle``, ``dataset_filters``,
            ``pending_regulator_pair``.

        """
        if pending_regulator_pair is not None and pending_regulator_pair() is not None:
            return True
        return (
            _pending_toggle() != _committed_toggle()
            or _pending_filters() != dataset_filters()
        )

    # Active dataset lists derived from committed toggle state.
    @reactive.calc
    def _active_binding_datasets() -> list[str]:
        """
        Binding datasets currently toggled on (committed state).

        :trigger: ``_committed_toggle`` — re-runs when committed state changes.

        """
        with perf(session.id, "select_datasets.sidebar", "_active_binding_datasets"):
            state = _committed_toggle()
            return [db for db, _, _ in binding_datasets if state.get(db, False)]

    @reactive.calc
    def _active_perturbation_datasets() -> list[str]:
        """
        Perturbation datasets currently toggled on (committed state).

        :trigger: ``_committed_toggle`` — re-runs when committed state changes.

        """
        with perf(
            session.id, "select_datasets.sidebar", "_active_perturbation_datasets"
        ):
            state = _committed_toggle()
            return [db for db, _, _ in perturbation_datasets if state.get(db, False)]

    # Instantiate one row sub-module per dataset. Each module owns the toggle
    # and filter-open effects for its row; all shared reactive state is passed
    # by reference so the row module can read and write it directly.
    @reactive.calc
    def _cached_meta_dfs() -> dict[str, pd.DataFrame]:
        """
        Unfiltered metadata DataFrames for all active datasets.

        Fetched once per active-dataset change and held in memory so that
        opening a filter modal does not trigger a redundant parquet scan.

        :trigger: ``_active_binding_datasets``, ``_active_perturbation_datasets`` —
            re-runs whenever the active dataset list changes.

        """
        # Gate vdb access until background materialization finishes; the DuckDB
        # connection is not safe to touch while materialization mutates it.
        if materialize_ready is not None:
            req(materialize_ready())
        all_active = _active_binding_datasets() + _active_perturbation_datasets()
        result: dict[str, pd.DataFrame] = {}
        for db in all_active:
            try:
                sql, params = metadata_query(db)
                result[db] = vdb.query(sql, **params)
            except Exception:
                logger.exception("Failed to fetch metadata for %s", db)
        return result

    @reactive.calc
    def _common_field_levels() -> dict[str, list[str]]:
        """
        Union of categorical levels for each common field across all active datasets.

        Derived from ``_cached_meta_dfs`` so no additional queries are issued.

        :trigger: ``_cached_meta_dfs`` — re-runs whenever the cached DataFrames change.

        """
        dfs = _cached_meta_dfs()
        result: dict[str, list[str]] = {}
        for cf_field in common_fields:
            levels: set[str] = set()
            for db, df in dfs.items():
                if cf_field not in df.columns:
                    continue
                col_dtype = df[cf_field].dtype
                type_override = FIELD_TYPE_OVERRIDES.get(
                    (db, cf_field)
                ) or FIELD_TYPE_OVERRIDES.get(("", cf_field))
                override_kind = type_override[0] if type_override else None
                if override_kind != "categorical" and col_dtype.name not in (
                    "object",
                    "category",
                ):
                    continue
                levels |= {str(v) for v in df[cf_field].dropna().unique()}
            if levels:
                result[cf_field] = list(levels)
        return result

    for db_name, _, _ in binding_datasets + perturbation_datasets:
        dataset_row_server(
            db_name,
            db_name=db_name,
            vdb=vdb,
            dataset_dict=dataset_dict,
            all_col_meta=all_col_meta,
            common_fields=common_fields,
            pending_toggle_state=_pending_toggle,
            pending_filters=_pending_filters,
            modal_open_for=modal_open_for,
            modal_df=modal_df,
            common_field_levels_fn=_common_field_levels,
            meta_dfs_fn=_cached_meta_dfs,
            upstream_cols=app_datasets.upstream_cols.get(db_name, []),
            modal_ns=session.ns,
            logger=logger,
        )

    # One-directional cascade: upstream categoricals (carbon source, temperature, etc.)
    # narrow the available condition checkbox choices. Condition selections do not feed
    # back into upstream selectizes — selecting additional conditions should expand (not
    # restrict) the available upstream values. Column classification is pre-computed in
    # app_datasets at startup; we only register the reactive effects here.
    for _db_name, _u_cols in app_datasets.upstream_cols.items():
        _cond_cols = app_datasets.condition_cols[_db_name]
        _db_meta = all_col_meta.get(_db_name, {})
        # Pre-compute all (column, input_id) pairs for this dataset so every
        # per-column cascade can read the combined state of all upstream filters.
        _all_u_col_id_pairs = [(col, f"filter_{_slugify(col)}") for col in _u_cols]

        for _upstream_col in _u_cols:
            _u_id = f"filter_{_slugify(_upstream_col)}"

            def _register_upstream_cascade(
                db_name: str,
                u_id: str,
                u_col: str,
                cond_cols: list[str],
                db_meta: dict[str, ColumnMeta],
                all_u_col_id_pairs: list[tuple[str, str]],
            ) -> None:
                """
                Register a cascade effect for one upstream column.

                All arguments are captured by value via the function signature so
                that each closure refers to the correct dataset and column names,
                not the loop variables at the time the effect fires.

                :param db_name: Dataset identifier — guards the effect so it only
                    runs when this dataset's modal is open.
                :param u_id: Shiny input ID of the upstream selectize widget.
                :param u_col: Column name in the metadata DataFrame that the
                    upstream selectize controls.
                :param cond_cols: Condition column names to update when the
                    upstream selection changes.
                :param db_meta: Per-column metadata for ``db_name``, used by
                    :func:`_build_experimental_condition_field_choices` to format
                    level labels.
                :param all_u_col_id_pairs: All ``(column, input_id)`` pairs for
                    every upstream column of this dataset. Used to build the
                    combined filter mask so that changes to one upstream selectize
                    always intersect with the current selections of the others.

                """

                @reactive.effect
                @reactive.event(input[u_id])
                def _cascade() -> None:
                    """
                    Narrow condition checkbox choices to levels that co-occur with the
                    intersection of all current categorical upstream selections. Only
                    updates ``choices``; the user's checkbox selection is preserved so
                    that previously checked conditions that are no longer valid are
                    removed without triggering a further cascade.

                    The mask is built across ALL categorical upstream columns
                    (not just the one that fired) so that datasets with a uniform
                    upstream column (e.g. every harbison row has Temperature = 37)
                    do not reset the narrowing done by a discriminating column
                    (e.g. Carbon source). Reads of other upstream inputs are safe
                    because ``@reactive.event`` isolates the entire body.

                    :trigger input[u_id]: fires when the upstream selectize changes.

                    """
                    with perf(session.id, "select_datasets.sidebar", "_cascade"):
                        if modal_open_for() != db_name:
                            return
                        df = modal_df()
                        if df is None or u_col not in df.columns:
                            return
                        # Cascade only applies when the triggering column is
                        # categorical. Numeric and boolean columns produce
                        # slider/switch values that cannot be used with isin()
                        # for range-aware filtering.
                        type_override = FIELD_TYPE_OVERRIDES.get(
                            (db_name, u_col)
                        ) or FIELD_TYPE_OVERRIDES.get(("", u_col))
                        override_kind = type_override[0] if type_override else None
                        col_dtype = df[u_col].dtype
                        is_categorical = (
                            override_kind == "categorical"
                            or col_dtype.name
                            in (
                                "object",
                                "category",
                            )
                        )
                        if not is_categorical:
                            return
                        # Build the combined mask across ALL categorical upstream
                        # columns. This prevents a race condition where separate
                        # per-column cascades fire in an unpredictable order and
                        # a uniform-valued column overwrites the narrowing applied
                        # by a discriminating one.
                        mask = pd.Series(True, index=df.index)
                        for _col, _uid in all_u_col_id_pairs:
                            if _col not in df.columns:
                                continue
                            _col_dtype = df[_col].dtype
                            _type_override = FIELD_TYPE_OVERRIDES.get(
                                (db_name, _col)
                            ) or FIELD_TYPE_OVERRIDES.get(("", _col))
                            _override_kind = (
                                _type_override[0] if _type_override else None
                            )
                            _is_cat = _override_kind == "categorical" or (
                                _col_dtype.name in ("object", "category")
                            )
                            if not _is_cat:
                                continue
                            try:
                                _sel = list(input[_uid]())
                            except SilentException:
                                _sel = []
                            if _sel:
                                mask &= df[_col].astype(str).isin(_sel)
                        for (
                            cond_col,
                            choices,
                        ) in _build_experimental_condition_field_choices(
                            df, mask, cond_cols, db_meta
                        ).items():
                            cond_id = f"filter_{_slugify(cond_col)}"
                            try:
                                cur = list(input[cond_id]())
                            except SilentException:
                                cur = list(choices)
                            ui.update_checkbox_group(
                                cond_id,
                                choices=choices,
                                selected=[v for v in cur if v in choices],
                            )

            _register_upstream_cascade(
                _db_name,
                _u_id,
                _upstream_col,
                _cond_cols,
                _db_meta,
                _all_u_col_id_pairs,
            )

    @reactive.effect
    @reactive.event(input.modal_reset_filters)
    def _reset_filter_modal() -> None:
        """
        Clear all filters for the open dataset (and common-field filters from every
        dataset), then close the modal.

        :trigger input.modal_reset_filters: fires when the user clicks the     Reset
        button inside the filter modal.

        """
        with perf(session.id, "select_datasets.sidebar", "_reset_filter_modal"):
            db_name = modal_open_for()
            if db_name is not None:
                current = dict(_pending_filters())
                all_db_names = [
                    d for d, _, _ in binding_datasets + perturbation_datasets
                ]
                # clear common-field filters from every dataset
                for ds in all_db_names:
                    if ds in current:
                        ds_filters = {
                            f: v
                            for f, v in current[ds].items()
                            if f not in common_fields
                        }
                        if ds_filters:
                            current[ds] = ds_filters
                        else:
                            current.pop(ds)
                # clear dataset-specific filters for the open dataset
                current.pop(db_name, None)
                _pending_filters.set(current)
                logger.debug(
                    "dataset_filters reset for %s: %d datasets with active filters",
                    db_name,
                    len(current),
                )
            ui.modal_remove()
            modal_open_for.set(None)
            modal_df.set(None)

    @reactive.effect
    @reactive.event(input.modal_clear_regulator_filter)
    def _clear_regulator_filter() -> None:
        """
        Remove ``regulator_locus_tag`` from all datasets and clear the selectize in the
        open modal in place via ``ui.update_selectize``.

        :trigger input.modal_clear_regulator_filter: fires when the user clicks the
        Clear button inside the Regulator card of a filter modal.

        """
        with perf(session.id, "select_datasets.sidebar", "_clear_regulator_filter"):
            db_name = modal_open_for()
            if db_name is None:
                return
            all_db_names = [d for d, _, _ in binding_datasets + perturbation_datasets]
            current = dict(_pending_filters())
            for ds in all_db_names:
                ds_filters = dict(current.get(ds, {}))
                ds_filters.pop("regulator_locus_tag", None)
                if ds_filters:
                    current[ds] = ds_filters
                else:
                    current.pop(ds, None)
            _pending_filters.set(current)
            # clear the selectize in place — no modal teardown/re-show needed
            ui.update_selectize("filter_regulator_locus_tag", selected=[])

    @reactive.effect
    @reactive.event(input.modal_apply_filters)
    def _apply_filter_modal() -> None:
        """
        Read filter inputs from the modal, persist them to ``dataset_filters``, activate
        the dataset if it was off, then close the modal.

        Common-field filters are propagated to all datasets or just this one
        according to each field's ``apply_to_all`` toggle.

        :trigger input.modal_apply_filters: fires when the user clicks the
            Apply Filters button inside the filter modal.

        """
        with perf(session.id, "select_datasets.sidebar", "_apply_filter_modal"):
            db_name = modal_open_for()
            df = modal_df()
            if db_name is None or df is None:
                ui.modal_remove()
                return

            field_filters: dict[str, Any] = {}
            for field in df.columns:
                if field == "sample_id":
                    continue

                col = df[field]
                try:
                    value = input[f"filter_{_slugify(field)}"]()
                except SilentException:
                    continue

                type_override = FIELD_TYPE_OVERRIDES.get(
                    (db_name, field)
                ) or FIELD_TYPE_OVERRIDES.get(("", field))
                override_kind = type_override[0] if type_override else None

                if override_kind == "categorical" or col.dtype.name in (
                    "object",
                    "category",
                ):
                    raw = list(value) if value else []
                    if raw:
                        # Coerce string selectize values back to the column's
                        # native numeric type so that = ANY(...) binds correctly
                        # against DOUBLE/INTEGER columns treated as categorical.
                        if col.dtype.name in ("float64", "float32"):
                            selected: list = [float(v) for v in raw]
                        elif col.dtype.name in ("int64", "int32"):
                            selected = [int(v) for v in raw]
                        else:
                            selected = raw
                        field_filters[field] = {
                            "type": "categorical",
                            "value": selected,
                        }

                elif col.dtype == "bool":
                    if bool(value):
                        field_filters[field] = {"type": "bool", "value": True}

                elif col.dtype.name in ("float64", "int64", "float32", "int32"):
                    if isinstance(value, (list, tuple)) and len(value) == 2:
                        non_null = col.dropna()
                        if non_null.empty:
                            continue
                        data_min = float(non_null.min())
                        data_max = float(non_null.max())
                        # single-value column: slider was artificially bumped in UI,
                        # user cannot meaningfully filter it — skip
                        if data_min == data_max:
                            continue
                        s_min, s_max = float(value[0]), float(value[1])
                        if s_min != data_min or s_max != data_max:
                            field_filters[field] = {
                                "type": "numeric",
                                "value": [s_min, s_max],
                            }

                # read per-field apply_to_all toggle for common fields
                if field in common_fields and field in field_filters:
                    try:
                        apply_to_all = bool(input[f"apply_to_all_{_slugify(field)}"]())
                    except SilentException:
                        apply_to_all = False
                    field_filters[field]["apply_to_all"] = apply_to_all

            # handle regulator_locus_tag explicitly (hidden from generic field loop)
            try:
                reg_selected = list(input["filter_regulator_locus_tag"]())
            except SilentException:
                reg_selected = []
            try:
                reg_apply_to_all = bool(input["apply_to_all_regulator_locus_tag"]())
            except SilentException:
                reg_apply_to_all = True
            if reg_selected:
                saved_reg = (
                    _pending_filters().get(db_name, {}).get("regulator_locus_tag", {})
                )
                from_pair = saved_reg.get("from_pair") if saved_reg else None
                reg_spec: dict[str, Any] = {
                    "type": "categorical",
                    "value": reg_selected,
                    "apply_to_all": reg_apply_to_all,
                }
                if from_pair:
                    reg_spec["from_pair"] = from_pair
                field_filters["regulator_locus_tag"] = reg_spec

            # split into common-field filters and dataset-specific
            # regulator_locus_tag is treated as a common field for propagation purposes
            reg_filter = field_filters.pop("regulator_locus_tag", None)
            common_filters = {
                f: v for f, v in field_filters.items() if f in common_fields
            }
            specific_filters = {
                f: v for f, v in field_filters.items() if f not in common_fields
            }

            current = dict(_pending_filters())
            all_db_names = [d for d, _, _ in binding_datasets + perturbation_datasets]

            # apply regulator filter (or clear it if empty)
            if reg_filter:
                if reg_filter.get("apply_to_all", True):
                    for ds in all_db_names:
                        ds_filters = dict(current.get(ds, {}))
                        ds_filters["regulator_locus_tag"] = reg_filter
                        current[ds] = ds_filters
                else:
                    for ds in all_db_names:
                        ds_filters = dict(current.get(ds, {}))
                        if ds == db_name:
                            ds_filters["regulator_locus_tag"] = reg_filter
                        else:
                            ds_filters.pop("regulator_locus_tag", None)
                        if ds_filters:
                            current[ds] = ds_filters
                        else:
                            current.pop(ds, None)
            else:
                # regulator field was cleared in the modal — remove from all datasets,
                # but only if the existing filter was set via the modal selectize.
                # A pairwise filter (from_pair_db) is committed via Apply Changes and
                # must not be wiped by opening an unrelated dataset's filter modal.
                existing_reg = current.get(db_name, {}).get("regulator_locus_tag", {})
                if not (existing_reg and existing_reg.get("from_pair_db")):
                    for ds in all_db_names:
                        ds_filters = dict(current.get(ds, {}))
                        ds_filters.pop("regulator_locus_tag", None)
                        if ds_filters:
                            current[ds] = ds_filters
                        else:
                            current.pop(ds, None)

            # apply each common filter according to its own apply_to_all flag
            for f, spec in common_filters.items():
                apply_to_all = spec.get("apply_to_all", True)
                if apply_to_all:
                    for ds in all_db_names:
                        ds_filters = dict(current.get(ds, {}))
                        ds_filters[f] = spec
                        current[ds] = ds_filters
                else:
                    # apply only to this dataset; clear from others
                    for ds in all_db_names:
                        ds_filters = dict(current.get(ds, {}))
                        if ds == db_name:
                            ds_filters[f] = spec
                        else:
                            ds_filters.pop(f, None)
                        if ds_filters:
                            current[ds] = ds_filters
                        else:
                            current.pop(ds, None)

            # clear common fields that were removed (not in common_filters)
            for f in common_fields:
                if f not in common_filters:
                    # check how this field was previously stored
                    # to decide scope of removal
                    prev_spec = current.get(db_name, {}).get(f)
                    prev_apply_to_all = (
                        prev_spec.get("apply_to_all", False) if prev_spec else False
                    )
                    targets = all_db_names if prev_apply_to_all else [db_name]
                    for ds in targets:
                        ds_filters = dict(current.get(ds, {}))
                        ds_filters.pop(f, None)
                        if ds_filters:
                            current[ds] = ds_filters
                        else:
                            current.pop(ds, None)

            # apply dataset-specific filters to just this dataset
            ds_filters = dict(current.get(db_name, {}))
            ds_filters.update(specific_filters)
            # remove any specific fields that are no longer set
            for f in list(ds_filters):
                if f not in common_fields and f not in specific_filters:
                    ds_filters.pop(f)
            if ds_filters:
                current[db_name] = ds_filters
            else:
                current.pop(db_name, None)

            _pending_filters.set(current)
            logger.debug(
                "dataset_filters (pending) applied for %s: %d fields set",
                db_name,
                len(ds_filters),
            )

            # activate the dataset in pending state if it isn't already on
            if not _pending_toggle().get(db_name, False):
                _pending_toggle.set({**_pending_toggle(), db_name: True})

            ui.modal_remove()
            modal_open_for.set(None)
            modal_df.set(None)

    @reactive.effect
    @reactive.event(input.apply_pending)
    def _apply_pending() -> None:
        """
        Commit pending toggle, filter, and regulator-pair state to the live reactive
        values, triggering the matrix and correlation queries exactly once.

        If a pairwise regulator filter is pending, its pre-computed intersection is
        merged into the filter state before committing. The pending pair is then
        cleared.

        :trigger input.apply_pending: fires when the user clicks the Apply button in the
        sidebar.

        """
        new_filters = dict(_pending_filters())

        if pending_regulator_pair is not None:
            pending = pending_regulator_pair()
            if pending is not None:
                db_a, db_b = pending["pair"]
                common_tags: list[str] = pending["common_tags"]
                display_a, display_b = pending["display"]
                if common_tags:
                    reg_spec: dict[str, Any] = {
                        "type": "categorical",
                        "value": common_tags,
                        "from_pair": (display_a, display_b),
                        "from_pair_db": (db_a, db_b),
                    }
                    for db_name in vdb.get_datasets():
                        ds_filters = dict(new_filters.get(db_name, {}))
                        ds_filters["regulator_locus_tag"] = reg_spec
                        new_filters[db_name] = ds_filters
                pending_regulator_pair.set(None)

        # Keep _pending_filters in sync with what we are about to commit so
        # that _has_pending_changes() returns False immediately after this call
        # and the Apply button deactivates.
        _pending_filters.set(new_filters)
        _committed_toggle.set(_pending_toggle())
        dataset_filters.set(new_filters)
        logger.debug(
            "_apply_pending committed: %d datasets active, %d datasets with filters",
            sum(_pending_toggle().values()),
            len(new_filters),
        )

    @render.ui
    def sidebar_content() -> ui.Tag:
        """
        Dataset rows for the selection sidebar.

        :trigger input.search: re-renders when the search input changes.
        :trigger _pending_filters: re-renders when filter state changes.

        Toggle state is read with ``reactive.isolate()`` so that toggling a
        dataset does NOT trigger a full sidebar re-render.  The
        ``ui.input_switch`` widget manages its own client-side state after
        initial render; programmatic state changes are synced via
        ``ui.update_switch`` in a separate effect in ``dataset_row_server``.

        """
        pending_toggle = _pending_toggle()
        active_filter_names: set[str] = {
            db for db in _pending_filters() if pending_toggle.get(db, False)
        }
        has_pending = _has_pending_changes()

        search_term = ""
        try:
            search_term = (input.search() or "").strip().lower()
        except SilentException:
            pass

        def _dataset_row(db_name: str, label: str, description: str) -> ui.Tag:
            with reactive.isolate():
                current_val = _pending_toggle().get(db_name, False)
            return dataset_row_ui(
                db_name,
                label=label,
                description=description,
                current_val=current_val,
                is_collapsed=False,
                has_active_filter=db_name in active_filter_names,
            )

        section_tags: list[ui.Tag] = []

        visible_binding = [
            (db_name, label, desc)
            for db_name, label, desc in binding_datasets
            if not search_term or search_term in label.lower()
        ]
        if visible_binding:
            section_tags.append(
                ui.div({"class": "group-header sidebar-text"}, "Binding")
            )
            for db_name, label, desc in visible_binding:
                section_tags.append(_dataset_row(db_name, label, desc))

        visible_perturbation = [
            (db_name, label, desc)
            for db_name, label, desc in perturbation_datasets
            if not search_term or search_term in label.lower()
        ]
        if visible_perturbation:
            section_tags.append(
                ui.div({"class": "group-header sidebar-text"}, "Perturbation")
            )
            for db_name, label, desc in visible_perturbation:
                section_tags.append(_dataset_row(db_name, label, desc))

        if not section_tags:
            section_tags.append(
                ui.div(
                    {"class": "empty-state compact"},
                    ui.p("No datasets match your search."),
                )
            )

        banner = (
            ui.div(
                {"class": "pending-banner"},
                "Dataset selection has changed. Click Apply Changes to update.",
            )
            if has_pending
            else ui.span()
        )
        return ui.div(
            ui.h2("Select datasets for analysis"),
            banner,
            ui.input_action_button(
                "apply_pending",
                "Apply Changes",
                class_="btn-apply-pending"
                + ("" if has_pending else " btn-apply-pending--idle"),
            ),
            ui.div({"class": "dataset-list"}, *section_tags),
        )

    return _active_binding_datasets, _active_perturbation_datasets, dataset_filters


__all__ = ["select_datasets_sidebar_server"]
