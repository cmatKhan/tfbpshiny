"""Sidebar server for the Select Datasets page."""

from __future__ import annotations

import asyncio
import io
from logging import Logger
from typing import Any

import faicons as fa
import pandas as pd
from labretriever import ColumnMeta, VirtualDB
from shiny import module, reactive, render, ui
from shiny.types import SilentException

from tfbpshiny.components import export_download_button
from tfbpshiny.modules.select_datasets.export import (
    ExportDataset,
    build_export_tarball,
)
from tfbpshiny.modules.select_datasets.queries import (
    FIELD_TYPE_OVERRIDES,
    full_data_query,
    metadata_query,
)
from tfbpshiny.modules.select_datasets.server.dataset_row import (
    dataset_row_server,
    dataset_row_ui,
)
from tfbpshiny.modules.select_datasets.ui import _slugify
from tfbpshiny.utils.vdb_init import (
    DEFAULT_ACTIVE_DATASETS,
    DEFAULT_DATASET_FILTERS,
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


@module.server
def select_datasets_sidebar_server(
    input: Any,
    output: Any,
    session: Any,
    vdb: VirtualDB,
    app_datasets: AppDatasets,
    logger: Logger,
    active_module: reactive.Value[str] | None = None,
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
    # Kept as a lookup map for display_name (used in export_datasets and
    # passed to dataset_row_server for the filter-modal title). The two
    # sorted lists below are derived views that provide rendering order and
    # the active-dataset calcs. They also carry description so it is
    # fetched once at startup rather than on every render.
    dataset_dict: dict[str, dict[str, str]] = {}
    for db_name in vdb.get_datasets():
        tags = vdb.get_tags(db_name)
        if tags.get("data_type") in ["binding", "perturbation"]:
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

    # reactives
    collapsed: reactive.Value[bool] = reactive.value(False)
    # {<db_name>: {<field_name>: {"type": "categorical" or "numeric" or "bool",
    #                              "value": list[str] | [lo, hi] | bool}}}
    dataset_filters: reactive.Value[dict[str, Any]] = reactive.value(
        DEFAULT_DATASET_FILTERS
    )
    # tracks which db_name's filter modal is currently open
    modal_open_for: reactive.Value[str | None] = reactive.value(None)
    # stores the DataFrame fetched when a filter modal is opened
    modal_df: reactive.Value[pd.DataFrame | None] = reactive.value(None)

    # Per-dataset toggle state — persists so toggles restore correctly on re-render.
    # Stored as a single reactive dict so all toggles are updated atomically.
    _toggle_state: reactive.Value[dict[str, bool]] = reactive.value(
        {
            db_name: db_name in DEFAULT_ACTIVE_DATASETS
            for db_name, _, _ in binding_datasets + perturbation_datasets
        }
    )

    # Active dataset lists derived from toggle state. Using @reactive.calc
    # instead of manually maintained reactive.Value eliminates redundant writes
    # and lets Shiny coalesce rapid toggle changes within a single flush cycle.
    @reactive.calc
    def _active_binding_datasets() -> list[str]:
        """
        Binding datasets currently toggled on.

        :trigger: ``_toggle_state`` — re-runs whenever any toggle changes.

        """
        state = _toggle_state()
        return [db for db, _, _ in binding_datasets if state.get(db, False)]

    @reactive.calc
    def _active_perturbation_datasets() -> list[str]:
        """
        Perturbation datasets currently toggled on.

        :trigger: ``_toggle_state`` — re-runs whenever any toggle changes.

        """
        state = _toggle_state()
        return [db for db, _, _ in perturbation_datasets if state.get(db, False)]

    @reactive.effect
    @reactive.event(input.toggle_sidebar)
    def _toggle_sidebar() -> None:
        """
        Toggle the sidebar between expanded and collapsed state.

        :trigger input.toggle_sidebar: fires when the user clicks the collapse/expand
        chevron button in the sidebar header.

        """
        collapsed.set(not collapsed())

    # Instantiate one row sub-module per dataset. Each module owns the toggle
    # and filter-open effects for its row; all shared reactive state is passed
    # by reference so the row module can read and write it directly.
    def _all_active() -> list[str]:
        return _active_binding_datasets() + _active_perturbation_datasets()

    for db_name, _, _ in binding_datasets + perturbation_datasets:
        dataset_row_server(
            db_name,
            db_name=db_name,
            vdb=vdb,
            dataset_dict=dataset_dict,
            all_col_meta=all_col_meta,
            common_fields=common_fields,
            toggle_state=_toggle_state,
            dataset_filters=dataset_filters,
            modal_open_for=modal_open_for,
            modal_df=modal_df,
            active_datasets_fn=_all_active,
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

        for _upstream_col in _u_cols:
            _u_id = f"filter_{_slugify(_upstream_col)}"

            def _register_upstream_cascade(
                db_name: str,
                u_id: str,
                u_col: str,
                cond_cols: list[str],
                db_meta: dict[str, ColumnMeta],
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

                """

                @reactive.effect
                @reactive.event(input[u_id])
                def _cascade() -> None:
                    """
                    Narrow condition checkbox choices to levels that co-occur with the
                    current upstream selection. Only updates ``choices``; the user's
                    checkbox selection is preserved so that previously checked
                    conditions that are no longer valid are removed without triggering a
                    further cascade.

                    :trigger input[u_id]: fires when the upstream selectize changes.

                    """
                    if modal_open_for() != db_name:
                        return
                    df = modal_df()
                    if df is None or u_col not in df.columns:
                        return
                    # Cascade only applies to categorical upstream columns.
                    # Numeric and boolean columns produce slider/switch values
                    # that cannot be used with isin() for range-aware filtering.
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
                    try:
                        sel = list(input[u_id]())
                    except SilentException:
                        sel = []
                    mask = (
                        df[u_col].isin(sel) if sel else pd.Series(True, index=df.index)
                    )
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
                _db_name, _u_id, _upstream_col, _cond_cols, _db_meta
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
        db_name = modal_open_for()
        if db_name is not None:
            current = dict(dataset_filters())
            all_db_names = [d for d, _, _ in binding_datasets + perturbation_datasets]
            # clear common-field filters from every dataset
            for ds in all_db_names:
                if ds in current:
                    ds_filters = {
                        f: v for f, v in current[ds].items() if f not in common_fields
                    }
                    if ds_filters:
                        current[ds] = ds_filters
                    else:
                        current.pop(ds)
            # clear dataset-specific filters for the open dataset
            current.pop(db_name, None)
            dataset_filters.set(current)
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
        db_name = modal_open_for()
        if db_name is None:
            return
        all_db_names = [d for d, _, _ in binding_datasets + perturbation_datasets]
        current = dict(dataset_filters())
        for ds in all_db_names:
            ds_filters = dict(current.get(ds, {}))
            ds_filters.pop("regulator_locus_tag", None)
            if ds_filters:
                current[ds] = ds_filters
            else:
                current.pop(ds, None)
        dataset_filters.set(current)
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
                selected = list(value) if value else []
                if selected:
                    field_filters[field] = {"type": "categorical", "value": selected}

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
                dataset_filters().get(db_name, {}).get("regulator_locus_tag", {})
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
        common_filters = {f: v for f, v in field_filters.items() if f in common_fields}
        specific_filters = {
            f: v for f, v in field_filters.items() if f not in common_fields
        }

        current = dict(dataset_filters())
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
            # regulator field was cleared — remove from all datasets
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
                # check how this field was previously stored to decide scope of removal
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

        dataset_filters.set(current)
        logger.debug(
            "dataset_filters applied for %s: %d fields set",
            db_name,
            len(ds_filters),
        )

        # activate the dataset if it isn't already on — the @reactive.calc
        # will automatically include it in the active list, and the row
        # module's _sync_toggle_to_dom effect will sync the DOM switch.
        if not _toggle_state().get(db_name, False):
            _toggle_state.set({**_toggle_state(), db_name: True})

        ui.modal_remove()
        modal_open_for.set(None)
        modal_df.set(None)

    @render.download(
        filename=lambda: "tfbpshiny_export.tar.gz",
        media_type="application/gzip",
    )
    async def export_datasets():
        """
        Build and stream a .tar.gz archive of all active datasets.

        The tarball is built in a worker thread via ``asyncio.to_thread`` so
        the Shiny event loop stays responsive.  A ``ui.Progress`` bar shows
        live per-dataset progress via an ``asyncio.Queue`` bridged from the
        worker thread with ``call_soon_threadsafe``.

        :trigger: ``input.export_datasets`` — fires when the user clicks the
            Export Selected Datasets download button.

        """
        all_active = _active_binding_datasets() + _active_perturbation_datasets()
        if not all_active:
            return

        filters = dataset_filters()
        n = len(all_active)

        # Build ExportDataset specs (SQL + params, not DataFrames)
        export_list: list[ExportDataset] = []
        for db_name in all_active:
            ds_filters = filters.get(db_name)
            display_name = dataset_dict[db_name].get("display_name", db_name)

            meta_sql, meta_params = metadata_query(db_name, ds_filters)
            data_sql, data_params = full_data_query(db_name, ds_filters)
            description = vdb.get_dataset_description(db_name)

            export_list.append(
                ExportDataset(
                    display_name=display_name,
                    metadata_sql=meta_sql,
                    metadata_params=meta_params,
                    data_sql=data_sql,
                    data_params=data_params,
                    description=description,
                )
            )

        # asyncio.Queue bridged from the worker thread for live progress.
        # A None sentinel signals that the build is complete.
        progress_q: asyncio.Queue[str | None] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def _on_dataset_done(name: str) -> None:
            loop.call_soon_threadsafe(progress_q.put_nowait, name)

        def _build_and_signal() -> io.BytesIO:
            try:
                return build_export_tarball(export_list, vdb, _on_dataset_done)
            finally:
                loop.call_soon_threadsafe(progress_q.put_nowait, None)

        with ui.Progress(min=0, max=n, session=session) as progress:
            progress.set(0, message="Preparing export...")

            build_task = asyncio.create_task(asyncio.to_thread(_build_and_signal))

            # Consume progress items until the sentinel arrives
            done = 0
            while True:
                name = await progress_q.get()
                if name is None:
                    break
                done += 1
                progress.set(
                    done,
                    message=f"Packaged {name}",
                    detail=f"{done} of {n}",
                )

            try:
                buf = await build_task
            except Exception:
                logger.exception("Export tarball build failed")
                return

            progress.set(n, message="Download ready")

        # Yield chunks from the in-memory buffer
        while chunk := buf.read(65536):
            yield chunk

    @render.ui
    def sidebar_panel() -> ui.Tag:
        """
        Full sidebar panel: header with collapse button, then Binding and
        Perturbation dataset rows with per-row toggles and Filter buttons.

        Toggle values are restored from ``_toggle_state`` so that re-renders
        (e.g. on navigation back to this page) reflect the current selection.

        :trigger collapsed: re-renders when the sidebar is collapsed or expanded.
        :trigger input.search: re-renders when the search input changes.

        Toggle state is read with ``reactive.isolate()`` so that toggling a
        dataset does NOT trigger a full sidebar re-render.  The
        ``ui.input_switch`` widget manages its own client-side state after
        initial render; programmatic state changes are synced via
        ``ui.update_switch`` in a separate effect.

        :trigger active_module: re-renders on page navigation so that
            toggle values are refreshed from ``_toggle_state`` when the
            user returns to the Select Datasets page.
        """
        # Read active_module to invalidate on navigation; the sidebar only
        # renders when active_module == "selection", but we re-compute on
        # every navigation change so toggle values are always fresh when the
        # user returns to this page.  Re-renders while the output element is
        # absent are deferred by Shiny until the element reappears.
        if active_module is not None:
            active_module()
        is_collapsed = collapsed()
        active_filter_names: set[str] = set(dataset_filters())

        search_term = ""
        if not is_collapsed:
            try:
                search_term = (input.search() or "").strip().lower()
            except SilentException:
                pass

        def _dataset_row(db_name: str, label: str, description: str) -> ui.Tag:
            # isolate: read toggle state without creating a reactive dependency —
            # toggles are synced by the row module's _sync_toggle_to_dom effect.
            with reactive.isolate():
                current_val = _toggle_state().get(db_name, False)
            return dataset_row_ui(
                db_name,
                label=label,
                description=description,
                current_val=current_val,
                is_collapsed=is_collapsed,
                has_active_filter=db_name in active_filter_names,
            )

        section_tags: list[ui.Tag] = []

        visible_binding = [
            (db_name, label, desc)
            for db_name, label, desc in binding_datasets
            if not search_term or search_term in label.lower()
        ]
        if visible_binding:
            if not is_collapsed:
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
            if not is_collapsed:
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

        return ui.div(
            {
                "class": "context-sidebar selection-sidebar"
                + (" collapsed" if is_collapsed else ""),
                "id": "selection-sidebar",
            },
            ui.div(
                {"class": "sidebar-header"},
                ui.div(
                    {"class": "sidebar-header-row"},
                    (
                        ui.div(
                            ui.h2("Select datasets\nfor analysis"),
                        )
                        if not is_collapsed
                        else ui.div(ui.h2("SD"))
                    ),
                    ui.input_action_button(
                        "toggle_sidebar",
                        (
                            fa.icon_svg("angles-left", width="14px", height="14px")
                            if not is_collapsed
                            else fa.icon_svg(
                                "angles-right", width="14px", height="14px"
                            )
                        ),
                        class_="btn-collapse-sidebar",
                    ),
                ),
            ),
            ui.div(
                {"class": "sidebar-body"},
                ui.div({"class": "dataset-list"}, *section_tags),
            ),
            ui.output_ui("sidebar_footer"),
        )

    @render.ui
    def sidebar_footer() -> ui.TagChild:
        """
        Export button footer — rendered independently so it can react to active-dataset
        changes without triggering a full sidebar re-render.

        :trigger: ``_active_binding_datasets``, ``_active_perturbation_datasets``,
            ``collapsed`` — re-renders when active datasets or collapse state change.

        """
        has_active = bool(_active_binding_datasets() or _active_perturbation_datasets())
        is_collapsed = collapsed()
        if has_active and not is_collapsed:
            return ui.div(
                {"class": "sidebar-footer"},
                export_download_button("export_datasets"),
            )
        return None

    return _active_binding_datasets, _active_perturbation_datasets, dataset_filters


__all__ = ["select_datasets_sidebar_server"]
