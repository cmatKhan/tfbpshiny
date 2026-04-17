"""Dataset-row sub-module for the Select Datasets sidebar."""

from __future__ import annotations

from collections.abc import Callable
from logging import Logger
from typing import Any

import pandas as pd
from labretriever import ColumnMeta, VirtualDB
from shiny import module, reactive, ui
from shiny.types import SilentException

from tfbpshiny import components
from tfbpshiny.modules.select_datasets.queries import (
    FIELD_TYPE_OVERRIDES,
    metadata_query,
    regulator_display_labels_query,
)
from tfbpshiny.modules.select_datasets.ui import (
    dataset_filter_modal_ui,
)
from tfbpshiny.utils.vdb_init import HIDDEN_FILTER_FIELDS


@module.ui
def dataset_row_ui(
    label: str,
    description: str,
    current_val: bool,
    is_collapsed: bool,
    has_active_filter: bool,
) -> ui.Tag:
    """
    Render one dataset row (toggle switch + optional filter button).

    :param label: Human-readable dataset name.
    :param description: Dataset description shown as a tooltip.
    :param current_val: Current toggle state.
    :param is_collapsed: When ``True``, renders only the toggle switch.
    :param has_active_filter: When ``True``, adds the active-filter CSS class to the
        filter button.

    """
    if is_collapsed:
        return ui.div(
            {"class": "dataset-row"},
            ui.input_switch("toggle", label=None, value=current_val),
        )
    label_span = ui.span({"class": "dataset-row-label sidebar-text"}, label)
    if description:
        label_span = components.tooltip(label_span, description, placement="right")
    return ui.div(
        {"class": "dataset-row"},
        ui.input_switch("toggle", label=label_span, value=current_val),
        ui.input_action_button(
            "filter_btn",
            "Filter",
            class_="btn-filter-dataset"
            + (" btn-filter-active" if has_active_filter else ""),
        ),
    )


@module.server
def dataset_row_server(
    input: Any,
    output: Any,
    session: Any,
    *,
    db_name: str,
    vdb: VirtualDB,
    dataset_dict: dict[str, dict[str, str]],
    all_col_meta: dict[str, dict[str, ColumnMeta]],
    common_fields: set[str],
    toggle_state: reactive.Value[dict[str, bool]],
    dataset_filters: reactive.Value[dict[str, Any]],
    modal_open_for: reactive.Value[str | None],
    modal_df: reactive.Value[pd.DataFrame | None],
    active_datasets_fn: Callable[[], list[str]],
    modal_ns: Callable[[str], str],
    logger: Logger,
) -> None:
    """
    Register reactive effects for one dataset row.

    Writes to the shared ``toggle_state``, ``dataset_filters``, ``modal_open_for``,
    and ``modal_df`` reactive values provided by the parent module. Does not return
    anything; all communication with the parent is through those shared values.

    :param db_name: Dataset identifier — used as guard and dict key.
    :param vdb: VirtualDB instance for runtime queries.
    :param dataset_dict: Mapping of ``db_name`` to tag dict from the parent.
    :param all_col_meta: Per-column metadata keyed by ``db_name``.
    :param common_fields: Field names shared across all datasets.
    :param toggle_state: Shared reactive dict of ``{db_name: bool}``.
    :param dataset_filters: Shared reactive dict of active filters.
    :param modal_open_for: Shared reactive tracking which dataset's modal is open.
    :param modal_df: Shared reactive holding the open modal's metadata DataFrame.
    :param active_datasets_fn: Callable that returns all currently active dataset
        names (binding + perturbation), used to compute common-field level unions.
    :param modal_ns: Namespace function from the parent module server
        (``session.ns``). Applied to all input IDs rendered inside the filter
        modal so they are registered under the parent's scope, not the row
        sub-module's scope.
    :param logger: Application logger.

    """

    @reactive.effect
    def _sync_toggle_to_dom() -> None:
        """
        Keep the DOM switch in sync with ``toggle_state`` so that programmatic
        activations (e.g. when the user applies filters to an off dataset) are reflected
        in the UI without requiring a full sidebar re-render.

        Uses a guard to avoid echoing back the value the user just set, which
        would trigger ``_on_toggle`` again unnecessarily.

        :trigger toggle_state: fires whenever any dataset's toggle changes.

        """
        val = toggle_state().get(db_name, False)
        with reactive.isolate():
            try:
                current = bool(input.toggle())
            except SilentException:
                return
        if current != val:
            ui.update_switch("toggle", value=val)

    @reactive.effect
    @reactive.event(input.toggle)
    def _on_toggle() -> None:
        """
        Update shared toggle state when the switch is flipped.

        Clears this dataset's filters when toggled off so the filter button
        returns to its inactive style and stale filters do not persist.

        The isolate guard prevents a reactive dependency on ``toggle_state``
        itself and avoids a redundant write when ``ui.update_switch`` echoes
        the same value back.

        :trigger input.toggle: fires when the user flips this dataset's switch.

        """
        try:
            val = bool(input.toggle())
        except SilentException:
            return
        with reactive.isolate():
            if toggle_state().get(db_name) == val:
                return
        toggle_state.set({**toggle_state(), db_name: val})
        if not val:
            current = dict(dataset_filters())
            current.pop(db_name, None)
            dataset_filters.set(current)

    @reactive.effect
    @reactive.event(input.filter_btn)
    def _open_filter_modal() -> None:
        """
        Fetch metadata and show the filter modal for this dataset.

        Builds the union of common-field levels across all active datasets so that valid
        values from other datasets remain selectable in the modal.

        :trigger input.filter_btn: fires when the user clicks the Filter button for this
        dataset.

        """
        existing_filters = dataset_filters().get(db_name)
        sql, params = metadata_query(db_name, existing_filters)
        df = vdb.query(sql, **params)
        modal_open_for.set(db_name)
        modal_df.set(df)
        display_name = dataset_dict[db_name].get("display_name", db_name)

        # build union of categorical levels for each common field across all
        # active datasets, so all valid values are selectable in the modal
        all_active = active_datasets_fn()
        common_field_levels: dict[str, list[str]] = {}
        for cf_field in common_fields:
            if cf_field not in df.columns:
                continue
            col_dtype = df[cf_field].dtype
            type_override = FIELD_TYPE_OVERRIDES.get(
                (db_name, cf_field)
            ) or FIELD_TYPE_OVERRIDES.get(("", cf_field))
            override_kind = type_override[0] if type_override else None
            if override_kind != "categorical" and col_dtype.name not in (
                "object",
                "category",
            ):
                continue
            levels: set[str] = {str(v) for v in df[cf_field].dropna().unique()}
            for other_db in all_active:
                if other_db == db_name:
                    continue
                try:
                    other_sql, other_params = metadata_query(other_db)
                    other_df = vdb.query(other_sql, **other_params)
                    if cf_field in other_df.columns:
                        levels |= {str(v) for v in other_df[cf_field].dropna().unique()}
                except Exception:
                    pass
            common_field_levels[cf_field] = list(levels)

        # build {locus_tag: "SYMBOL (LOCUS_TAG)"} map for regulator selectize
        reg_display_labels: dict[str, str] = {}
        try:
            reg_sql, reg_params = regulator_display_labels_query(db_name)
            reg_df = vdb.query(reg_sql, **reg_params)
            for _, row in reg_df.iterrows():
                tag = str(row["regulator_locus_tag"])
                sym = row.get("regulator_symbol")
                label = f"{sym} ({tag})" if sym and str(sym) != "nan" else tag
                reg_display_labels[tag] = label
        except Exception:
            logger.exception("Failed to fetch regulator display labels for %s", db_name)

        ui.modal_show(
            dataset_filter_modal_ui(
                db_name,
                df,
                existing_filters,
                common_fields,
                display_name=display_name,
                common_field_levels=common_field_levels,
                hidden_fields=HIDDEN_FILTER_FIELDS.get("*", set())
                | HIDDEN_FILTER_FIELDS.get(db_name, set()),
                regulator_display_labels=reg_display_labels or None,
                col_meta=all_col_meta.get(db_name) or None,
                ns=modal_ns,
            )
        )


__all__ = ["dataset_row_ui", "dataset_row_server"]
