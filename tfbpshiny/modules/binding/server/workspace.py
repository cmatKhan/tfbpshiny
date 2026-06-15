from __future__ import annotations

import asyncio
import itertools
from collections.abc import Callable
from logging import Logger
from typing import Any, Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from labretriever import VirtualDB
from plotly.io import to_html
from shiny import reactive, render, req, ui
from shiny.reactive import extended_task
from shiny.ui import (  # noqa: F401 (bind_task_button used as decorator)
    bind_task_button,
    input_task_button,
)
from shinywidgets import output_widget, render_plotly

from tfbpshiny.modules.binding.queries import (
    LOG10P_FLOOR,
    corr_all_pairs_sql,
    get_log10p_source,
    get_measurement_column,
    regulator_scatter_sql,
)
from tfbpshiny.utils.correlation_matrix import build_correlation_matrix_ui
from tfbpshiny.utils.perf import perf, reset_render_counts
from tfbpshiny.utils.vdb_init import AppDatasets, get_regulator_display_name


def binding_workspace_server(
    input: Any,
    output: Any,
    session: Any,
    active_binding_datasets: reactive.Calc_[list[str]],
    dataset_filters: reactive.Value[dict[str, Any]],
    vdb: VirtualDB,
    app_datasets: AppDatasets,
    logger: Logger,
    active_tab: reactive.Calc_[str] | None = None,
    materialize_ready: Callable[[], bool] | None = None,
) -> None:
    """
    Render the binding correlation workspace: correlation matrix, pair distribution
    box plot, and per-regulator scatter plots, gated on an explicit Execute Analysis
    button.

    :param active_binding_datasets: Reactive calc returning the list of active binding
        dataset names from the select-datasets module.
    :param dataset_filters: Reactive value holding the current filter state.
    :param vdb: Application VirtualDB instance.
    :param app_datasets: App-level dataset metadata (condition columns, etc.).
    :param logger: Application logger.
    :param active_tab: Optional reactive calc for the currently active tab name.

    """

    session.on_flush(lambda: reset_render_counts(session.id))

    display_names: dict[str, str] = {
        db_name: vdb.get_tags(db_name).get("display_name", db_name)
        for db_name in vdb.get_datasets()
    }

    _reg_df = get_regulator_display_name(vdb)
    sym_map: dict[str, str] = dict(
        zip(_reg_df["regulator_locus_tag"], _reg_df["display_name"])
    )

    # Currently selected regulator locus tag — shared across box and scatter renders.
    selected_reg: reactive.Value[str] = reactive.value("")

    # pending_pairs: toggled immediately by matrix cell clicks; drives the matrix
    # highlight only.  committed_pairs: set when Execute Analysis is clicked;
    # drives box plots and scatter plots.  Separating them means clicking cells is
    # instant (no expensive renders triggered) and Execute gates all heavy work.
    pending_pairs: reactive.Value[list[tuple[str, str]]] = reactive.value([])
    committed_pairs: reactive.Value[list[tuple[str, str]]] = reactive.value([])

    # Scatter render-phase tracking — resets when selected_reg changes so the
    # "preparing" message reappears for each new regulator.
    # _scatter_epoch is bumped on each reset; each _scatter_plot closure captures the
    # epoch at render-start so stale completions from a previous reg don't count.
    _scatter_epoch: reactive.Value[int] = reactive.value(0)
    _scatter_rendered: reactive.Value[int] = reactive.value(0)
    _scatter_expected: reactive.Value[int] = reactive.value(0)

    # Dicts keyed by pair for the active box FigureWidgets and their backing data.
    # Mutated in-place by the box render factory and _highlight_one.
    _pair_box_widgets: dict[tuple[str, str], go.FigureWidget | None] = {}
    _pair_box_data: dict[tuple[str, str], dict] = {}

    # Stable pair list — updated only when the active dataset set actually changes.
    _active_pairs: reactive.Value[list[tuple[str, str]]] = reactive.value([])

    # Snapshot of sidebar inputs at the time of the last Execute click.
    # None means Execute has never been clicked; always "pending" in that state.
    _last_run_snapshot: reactive.Value[tuple | None] = reactive.value(None)

    def _snapshot_current() -> tuple:
        """Return a hashable representation of the current sidebar inputs."""
        try:
            included: tuple = tuple(sorted(input.included_datasets() or ()))
        except Exception:
            included = ()
        try:
            filters_repr = repr(
                sorted((k, repr(v)) for k, v in dataset_filters().items())
            )
        except Exception:
            filters_repr = ""
        return (
            tuple(sorted(_active_pairs())),
            input.col_preference(),
            input.corr_type(),
            included,
            filters_repr,
            tuple(sorted(pending_pairs())),
        )

    @reactive.effect
    def _sync_pairs() -> None:
        """
        Write the pair list to ``_active_pairs`` only when it actually changes.

        :trigger active_binding_datasets: re-fires when the dataset selection changes.
        :trigger active_tab: silently blocks when another tab is active.

        """
        if active_tab is not None:
            req(active_tab() == "Binding")
        active = active_binding_datasets()
        new = list(itertools.combinations(sorted(active), 2))
        with reactive.isolate():
            if new != _active_pairs():
                _active_pairs.set(new)

    @render.ui
    def execute_pending_style() -> ui.Tag:
        """
        Inject a ``<style>`` tag that dims the Execute Analysis button when the current
        sidebar state matches the last-run snapshot (no pending changes).

        Using a server-rendered style tag (ID selector) rather than toggling a CSS class
        via JS ensures the dimming survives the bslib-task-button web component's own
        state transitions, which may reset class attributes.

        :trigger _active_pairs: re-fires when dataset selection changes. :trigger
        input.col_preference: re-fires on column change. :trigger input.corr_type: re-
        fires on correlation-method change. :trigger _last_run_snapshot: re-fires after
        Execute to reset the indicator. :trigger dataset_filters: re-fires when filters
        change. :trigger input.included_datasets: re-fires when dataset checkboxes
        change.

        """
        current = _snapshot_current()
        last = _last_run_snapshot()
        has_pending = (last is None) or (current != last)
        btn_id = session.ns("execute_analysis")
        if has_pending:
            return ui.div(
                {"class": "pending-banner"},
                "Analysis options have changed. Click Execute Analysis to apply.",
            )
        return ui.tags.style(
            f"#{btn_id} {{ opacity: 0.35; pointer-events: none; cursor: not-allowed; }}"
        )

    # --- Execute Analysis task --------------------------------------------------

    @bind_task_button(button_id="execute_analysis")
    @extended_task
    async def _run_analysis(
        pairs: list[tuple[str, str]],
        col_map: dict[str, str],
        col_preference: str,
        filters: dict,
        method: str,
    ) -> dict:
        """
        Compute per-regulator pairwise correlations off the main thread.

        :param pairs: Dataset pairs to compute.
        :param col_map: Mapping from db_name to measurement column name.
        :param col_preference: User's column preference (``"effect"``, ``"pvalue"``,
            or ``"log10pval"``).
        :param filters: Active dataset filters at execute time.
        :param method: Correlation method (``"pearson"`` or ``"spearman"``).
        :returns: Dict with keys ``corr_data``, ``pairs``, ``col_map``,
            ``col_preference``, ``method``, and ``filters``.

        """
        empty_cols = [
            "db_a",
            "db_a_id",
            "db_b",
            "db_b_id",
            "regulator_locus_tag",
            "correlation",
        ]
        if not pairs:
            return {
                "corr_data": {},
                "pairs": [],
                "col_map": col_map,
                "col_preference": col_preference,
                "method": method,
                "filters": filters,
            }

        try:
            with perf(
                session.id, "binding.workspace", "corr_all_pairs_sql", kind="data"
            ):
                combined = await asyncio.to_thread(
                    corr_all_pairs_sql, vdb, pairs, col_map, filters, method
                )
        except Exception as exc:
            logger.error("corr_all_pairs_sql failed: %s", exc, exc_info=True)
            combined = pd.DataFrame(columns=empty_cols + ["pair_key"])

        corr_data: dict[tuple[str, str], pd.DataFrame] = {}
        for db_a, db_b in pairs:
            key = f"{db_a}__{db_b}"
            if combined.empty or "pair_key" not in combined.columns:
                corr_data[(db_a, db_b)] = pd.DataFrame(columns=empty_cols)
            else:
                subset = (
                    combined[combined["pair_key"] == key]
                    .drop(columns=["pair_key"])
                    .reset_index(drop=True)
                )
                corr_data[(db_a, db_b)] = subset

        return {
            "corr_data": corr_data,
            "pairs": pairs,
            "col_map": col_map,
            "col_preference": col_preference,
            "method": method,
            "filters": filters,
        }

    @reactive.effect
    @reactive.event(input.execute_analysis)
    def _on_execute() -> None:
        """
        Read current sidebar state and invoke the analysis task.

        :trigger input.execute_analysis: fires when the Execute Analysis button is
        clicked.

        """
        # Gate vdb access until background materialization finishes; the DuckDB
        # connection is not safe to touch while materialization mutates it.
        if materialize_ready is not None:
            req(materialize_ready())
        with perf(session.id, "binding.workspace", "_on_execute"):
            pairs = _active_pairs()
            method = input.corr_type()
            preference: Literal["effect", "pvalue", "log10pval"] = (
                input.col_preference()  # type: ignore[assignment]
            )
            filters = dataset_filters()

            # Restrict to datasets checked in the sidebar checkbox group.
            try:
                included = set(input.included_datasets())
            except Exception:
                included = {db for p in pairs for db in p}

            pairs = [(a, b) for a, b in pairs if a in included and b in included]
            col_map = {
                db: get_measurement_column(db, preference)
                for pair in pairs
                for db in pair
            }
            # Scatter counters are reset here so analysis_status immediately shows
            # "preparing" on the scatter tab; _init_selected_reg or
            # _reset_scatter_epoch will set _scatter_expected once the task succeeds.
            _scatter_rendered.set(0)
            _scatter_expected.set(0)
            # Commit the pending pair selection so box plots and scatters update.
            with reactive.isolate():
                committed_pairs.set(list(pending_pairs()))
            _run_analysis.invoke(pairs, col_map, preference, filters, method)
            # Capture snapshot so execute_pending_style dims the button.
            _last_run_snapshot.set(_snapshot_current())

    # --- Helpers ----------------------------------------------------------------

    def _visible_scatter_count(
        pairs: list[tuple[str, str]], sel: list[tuple[str, str]]
    ) -> int:
        """
        Return the number of active pairs that will have scatter slots emitted, matching
        the filter logic in ``scatter_container``.

        :param pairs: Active pairs from the task result.
        :param sel: Currently selected pairs.

        """
        if not sel:
            return len(pairs)
        sel_dbs: set[str] = {db for p in sel for db in p}
        return sum(1 for p in pairs if set(p) <= sel_dbs)

    # --- Eager regulator and pair initialization --------------------------------

    @reactive.effect
    def _init_selected_reg() -> None:
        """
        Set ``selected_reg`` as soon as the task succeeds so scatter plots start
        computing immediately after the matrix renders.

        Preserves the current selection if it is still valid in the new result;
        falls back to the alphabetically first entry otherwise.  When the selection
        is already valid (no change to ``selected_reg``), the scatter epoch is bumped
        here directly because ``_reset_scatter_epoch`` only fires on reg *changes*.

        :trigger _run_analysis.status: fires when the task state changes to success.

        """
        if _run_analysis.status() != "success":
            return
        result = _run_analysis.result()
        all_regs: set[str] = set()
        for df in result["corr_data"].values():
            if not df.empty:
                all_regs |= set(df["regulator_locus_tag"].dropna().unique())
        if not all_regs:
            return
        choices = dict(
            sorted(
                {r: sym_map.get(r, r) for r in all_regs}.items(),
                key=lambda kv: kv[1].lower(),
            )
        )
        default = next(iter(choices))
        with reactive.isolate():
            cur = selected_reg()
        if cur not in choices:
            # Changing selected_reg will trigger _reset_scatter_epoch automatically.
            selected_reg.set(default)
        else:
            # Reg is unchanged; manually bump the epoch so the "preparing" message
            # fires and scatter plots start counting from zero for this run.
            with reactive.isolate():
                sel = committed_pairs()
                _scatter_epoch.set(_scatter_epoch() + 1)
                _scatter_rendered.set(0)
                _scatter_expected.set(_visible_scatter_count(result["pairs"], sel))

    @reactive.effect
    def _init_selected_pairs() -> None:
        """
        Prune stale pairs from ``pending_pairs`` and ``committed_pairs`` when the task
        succeeds.

        No default pair is seeded: the matrix selection is the single source of
        truth, so the distribution and scatter stay empty until the user clicks a
        matrix cell (and re-runs Execute Analysis to commit the selection).

        :trigger _run_analysis.status: fires when the task transitions to success.

        """
        if _run_analysis.status() != "success":
            return
        pairs = _run_analysis.result()["pairs"]
        pairs_set = set(pairs)
        with reactive.isolate():
            cur_pending = pending_pairs()
            cur_committed = committed_pairs()
        valid_pending = [p for p in cur_pending if p in pairs_set]
        valid_committed = [p for p in cur_committed if p in pairs_set]
        if valid_pending != cur_pending:
            pending_pairs.set(valid_pending)
        if valid_committed != cur_committed:
            committed_pairs.set(valid_committed)

    @reactive.effect
    def _reset_scatter_epoch() -> None:
        """
        Bump the scatter render epoch whenever ``selected_reg`` changes so stale scatter
        renders from the previous regulator don't increment the current epoch's counter.

        ``committed_pairs`` is read under ``reactive.isolate`` so that
        ``_init_selected_pairs`` setting it on task success does not re-trigger
        this effect and spuriously bump the epoch after renders have already
        started counting.

        :trigger selected_reg: fires when the user selects a different regulator.

        """
        reg = selected_reg()
        if not reg:
            return
        with reactive.isolate():
            if _run_analysis.status() != "success":
                return
            result = _run_analysis.result()
            sel = committed_pairs()
            _scatter_epoch.set(_scatter_epoch() + 1)
            _scatter_rendered.set(0)
            _scatter_expected.set(_visible_scatter_count(result["pairs"], sel))

    # --- Status render ----------------------------------------------------------

    @render.ui
    def analysis_status() -> ui.Tag:
        """
        User feedback while the task is running or has errored.

        :trigger _run_analysis.status: re-renders when the task state changes.

        """
        status = _run_analysis.status()
        if status == "running":
            return ui.div(
                {"class": "empty-state"},
                ui.p(
                    "Computing correlations. This typically takes less than 5 seconds."
                    " Thank you for your patience."
                ),
            )
        if status == "error":
            return ui.div(
                {"class": "empty-state"},
                ui.p(f"Error: {_run_analysis.error()}"),
            )
        return ui.span()

    # --- Dataset selection sidebar render ---------------------------------------

    @render.ui
    def dataset_selection() -> ui.Tag:
        """
        Checkbox group for including/excluding active binding datasets.

        :trigger active_binding_datasets: re-renders when the dataset set changes.

        """
        datasets = active_binding_datasets()
        if not datasets:
            return ui.span()
        return ui.input_checkbox_group(
            "included_datasets",
            label=None,
            choices={db: display_names.get(db, db) for db in sorted(datasets)},
            selected=datasets,
        )

    # --- Box plot helpers -------------------------------------------------------

    def _highlight_one(pair: tuple[str, str], reg: str) -> None:
        """
        Mutate trace 1 of the box FigureWidget for ``pair`` in-place to highlight
        ``reg``'s point.

        Sends a ``_py2js_restyle`` delta to the client via ``batch_update``; the
        box trace (trace 0) is never touched, so no full figure re-render occurs.

        :param pair: Dataset pair key into ``_pair_box_widgets`` / ``_pair_box_data``.
        :param reg: Regulator locus tag to highlight, or ``""`` to clear.

        """
        fig = _pair_box_widgets.get(pair)
        data = _pair_box_data.get(pair)
        if fig is None:
            return
        if not reg or data is None:
            with fig.batch_update():
                fig.data[1].x = []
                fig.data[1].y = []
                fig.data[1].hovertext = []
            return
        all_x = data["all_x"]
        all_y = data["all_y"]
        all_tags = data["all_tags"]
        all_hover = data["all_hover"]
        idx = [i for i, t in enumerate(all_tags) if t == reg]
        sel_x = [all_x[i] for i in idx]
        sel_y = [all_y[i] for i in idx]
        sel_hover = [all_hover[i] for i in idx]
        with fig.batch_update():
            fig.data[1].x = sel_x
            fig.data[1].y = sel_y
            fig.data[1].hovertext = sel_hover

    @reactive.effect
    def _update_all_highlights() -> None:
        """
        In-place update of the highlight trace in every live box FigureWidget.

        Fires when ``selected_reg`` changes. Never calls any render function;
        only the delta for trace 1 is sent to each widget.

        :trigger selected_reg: fires when the user clicks a point or picks
            from dropdown.

        """
        reg = selected_reg()
        for pair, fig in _pair_box_widgets.items():
            if fig is not None:
                _highlight_one(pair, reg)

    def _apply_log10p_transform(
        series: pd.Series,
        db_name: str,
        col: str,
        display: str,
    ) -> tuple[pd.Series, str]:
        """
        Apply the -log10 transform appropriate for the dataset's p-value source.

        Returns the transformed series and an axis label string.

        :param series: Raw column values from the query.
        :param db_name: Dataset name (used to look up the source type).
        :param col: Column name (used in the fallback axis label).
        :param display: Dataset display name for the axis label.

        """
        cap = -np.log10(LOG10P_FLOOR)  # = 10
        source = get_log10p_source(db_name)
        if source == "neglog10p":
            transformed = series.clip(upper=cap)
            label = f"{display}: -log10(p)"
        elif source == "log10p":
            transformed = (-series).clip(upper=cap)
            label = f"{display}: -log10(p)"
        elif source == "pval":
            transformed = -np.log10(series.clip(lower=LOG10P_FLOOR))
            label = f"{display}: -log10(p)"
        else:
            transformed = series
            label = f"{display}: {col}"
        return transformed, label

    # --- All possible pairs (fixed at init) ------------------------------------

    _all_possible_pairs: list[tuple[str, str]] = list(
        itertools.combinations(
            sorted(
                db
                for db in vdb.get_datasets()
                if vdb.get_tags(db).get("data_type") == "binding"
            ),
            2,
        )
    )

    # --- Correlation matrix ----------------------------------------------------

    @output(suspend_when_hidden=False)
    @render.ui
    def corr_matrix_container() -> ui.Tag:
        """
        N×N correlation matrix table showing median r per dataset pair.

        Calls :func:`~tfbpshiny.utils.correlation_matrix.build_correlation_matrix_ui`
        to produce the table tag.  Re-renders when the task result changes or when
        ``selected_pair`` changes (to move the active-cell highlight).

        :trigger _run_analysis.status: re-renders when the task completes.
        :trigger selected_pair: re-renders to update the active-cell highlight.

        """
        with perf(session.id, "binding.workspace", "corr_matrix_container"):
            status = _run_analysis.status()
            if status != "success":
                if status == "initial":
                    return ui.div(
                        {"class": "empty-state"},
                        ui.p(
                            "Click Execute Analysis to compute pairwise binding "
                            "correlations."
                        ),
                    )
                return ui.span()

            result = _run_analysis.result()
            pairs: list[tuple[str, str]] = result["pairs"]
            if not pairs:
                return ui.div(
                    {"class": "empty-state"},
                    ui.p(
                        "No active binding dataset pairs. Select at least two datasets."
                    ),
                )

            with reactive.isolate():
                active_datasets = active_binding_datasets()

            return build_correlation_matrix_ui(
                all_possible_pairs=_all_possible_pairs,
                active_pairs=pairs,
                active_datasets=sorted(active_datasets),
                corr_data=result["corr_data"],
                display_names=display_names,
                selected_pairs=set(pending_pairs()),
                ns=session.ns,
            )

    # --- Cell click factory — pre-register one toggle effect per possible pair -

    def _make_cell_click_effect(db_a: str, db_b: str) -> None:
        """
        Pre-register the reactive effect that toggles ``(db_a, db_b)`` in
        ``selected_pairs`` when the corresponding matrix cell button is clicked.

        The button ID ``corrpair_{db_a}__{db_b}`` matches the ID emitted by
        :func:`~tfbpshiny.utils.correlation_matrix.build_correlation_matrix_ui`.

        :param db_a: First dataset name (canonical order from ``_all_possible_pairs``).
        :param db_b: Second dataset name.

        """
        btn_id = f"corrpair_{db_a}__{db_b}"
        pair = (db_a, db_b)

        @reactive.effect
        @reactive.event(input[btn_id])
        def _on_cell_click() -> None:
            with reactive.isolate():
                cur = list(pending_pairs())
            if pair in cur:
                cur.remove(pair)
            else:
                cur.append(pair)
            pending_pairs.set(cur)

    for _db_a, _db_b in _all_possible_pairs:
        _make_cell_click_effect(_db_a, _db_b)

    # --- Pair distribution box plots -------------------------------------------

    @output(suspend_when_hidden=False)
    @render.ui
    def pair_box_status() -> ui.Tag:
        """
        Shown when the task has succeeded but no pairs are selected yet.

        :trigger _run_analysis.status: re-renders when the task completes. :trigger
        selected_pairs: re-renders when the selection changes.

        """
        if _run_analysis.status() != "success":
            return ui.span()
        if not committed_pairs():
            return ui.div(
                {"class": "empty-state"},
                ui.p(
                    "Select cells in the Correlation Matrix and click Execute Analysis "
                    "to view their distributions."
                ),
            )
        return ui.span()

    @output(suspend_when_hidden=False)
    @render.ui
    def pair_box_container() -> ui.Tag:
        """
        Flex container of one ``output_widget`` slot per selected pair.

        Each slot is matched by a ``render_plotly`` registered by
        ``_make_pair_box_render`` at server start.

        :trigger selected_pairs: re-renders when the selection changes.
        :trigger _run_analysis.status: re-renders on task completion.

        """
        # Unmount the plotly figures when the Binding tab is not active so they
        # do not stay resident in the DOM (large figures across all tabs freeze
        # the browser). They re-render when the tab becomes active again.
        if active_tab is not None and active_tab() != "Binding":
            return ui.span()
        with perf(session.id, "binding.workspace", "pair_box_container"):
            if _run_analysis.status() != "success":
                return ui.span()
            pairs = committed_pairs()
            if not pairs:
                return ui.span()
            result = _run_analysis.result()
            active_pair_set = set(result["pairs"])
            slots = [
                ui.div(
                    output_widget(f"pair_box_{db_a}__{db_b}"), style="flex: 0 0 auto;"
                )
                for db_a, db_b in _all_possible_pairs
                if (db_a, db_b) in active_pair_set and (db_a, db_b) in set(pairs)
            ]
            if not slots:
                return ui.span()
            return ui.div(
                *slots,
                style=(
                    "display: flex; flex-wrap: wrap;"
                    " gap: 1rem; align-items: flex-start;"
                ),
            )

    def _make_pair_box_render(db_a: str, db_b: str) -> None:
        """
        Register a ``render_plotly`` for one dataset pair's box plot.

        The FigureWidget is stored in ``_pair_box_widgets`` so
        ``_update_all_highlights`` can mutate it in-place when ``selected_reg``
        changes without triggering a re-render.

        :param db_a: First dataset name.
        :param db_b: Second dataset name.

        """
        pair = (db_a, db_b)

        @output(id=f"pair_box_{db_a}__{db_b}")
        @render_plotly
        def _pair_box() -> go.FigureWidget:
            """
            Box + jittered-points for one selected pair's per-regulator correlations.

            Two-trace pattern: trace 0 is the box, trace 1 is the highlight overlay
            updated in-place by ``_update_all_highlights``.

            :trigger selected_pairs: re-renders when this pair enters the selection.
            :trigger _run_analysis.status: re-renders on task completion.

            """
            with perf(session.id, "binding.workspace", f"pair_box_{db_a}__{db_b}"):
                if _run_analysis.status() != "success":
                    _pair_box_widgets[pair] = None
                    return go.FigureWidget()

                with reactive.isolate():
                    sel = committed_pairs()
                if pair not in sel:
                    _pair_box_widgets[pair] = None
                    return go.FigureWidget()

                result = _run_analysis.result()
                if pair not in result["pairs"]:
                    _pair_box_widgets[pair] = None
                    return go.FigureWidget()

                df = result["corr_data"].get(pair, pd.DataFrame())
                method = result["method"]
                label_a = display_names.get(db_a, db_a)
                label_b = display_names.get(db_b, db_b)

                all_x: list[str] = []
                all_y: list[float] = []
                all_tags: list[str] = []
                all_hover: list[str] = []

                if not df.empty:
                    df_clean = df.dropna(subset=["correlation"])
                    for tag, corr in zip(
                        df_clean["regulator_locus_tag"],
                        df_clean["correlation"],
                    ):
                        all_x.append(f"{label_a}\nvs\n{label_b}")
                        all_y.append(float(corr))
                        all_tags.append(tag)
                        all_hover.append(sym_map.get(tag, tag))

                fig = go.FigureWidget()
                fig.add_trace(
                    go.Box(
                        x=all_x,
                        y=all_y,
                        text=all_hover,
                        customdata=all_tags,
                        hovertemplate="%{text}<extra></extra>",
                        hoveron="points",
                        boxpoints="all",
                        jitter=0.4,
                        pointpos=0,
                        marker=dict(size=4, opacity=0.5),
                        line=dict(width=1.5),
                        showlegend=False,
                    )
                )
                # Trace 1: highlight overlay — mutated in-place by _highlight_one.
                fig.add_trace(
                    go.Scatter(
                        x=[],
                        y=[],
                        mode="markers",
                        hovertext=[],
                        hovertemplate="%{hovertext}<extra></extra>",
                        marker=dict(size=10, color="black", symbol="circle"),
                        showlegend=False,
                    )
                )

                fig.update_layout(
                    title=dict(
                        text=f"{label_a}<br>vs<br>{label_b}", x=0.5, xanchor="center"
                    ),
                    yaxis=dict(title=f"{method.capitalize()} r", range=[-1, 1]),
                    showlegend=False,
                    margin=dict(l=50, r=20, t=100, b=60),
                    width=480,
                    height=460,
                )

                _pair_box_data[pair] = {
                    "all_x": all_x,
                    "all_y": all_y,
                    "all_tags": all_tags,
                    "all_hover": all_hover,
                }
                _pair_box_widgets[pair] = fig

                def _on_click(trace: Any, points: Any, state: Any) -> None:
                    if not points.point_inds:
                        return
                    selected_reg.set(all_tags[points.point_inds[0]])

                fig.data[0].on_click(_on_click)

                with reactive.isolate():
                    cur = selected_reg()
                if cur:
                    _highlight_one(pair, cur)

                return fig

    for _db_a, _db_b in _all_possible_pairs:
        _make_pair_box_render(_db_a, _db_b)

    # --- Regulator selectors (one per tab, linked via selected_reg) ------------

    def _reg_selector_choices() -> dict[str, str] | None:
        """
        Build the sorted choices dict for the regulator selectize inputs.

        Returns ``None`` when the task has not succeeded or no regulators exist.

        """
        if _run_analysis.status() != "success":
            return None
        all_regs: set[str] = set()
        for df in _run_analysis.result()["corr_data"].values():
            if not df.empty:
                all_regs |= set(df["regulator_locus_tag"].dropna().unique())
        if not all_regs:
            return None
        choices = {r: sym_map.get(r, r) for r in all_regs}
        return dict(sorted(choices.items(), key=lambda kv: kv[1].lower()))

    def _reg_selector_tag(input_id: str) -> ui.Tag:
        """
        Render a regulator selectize input with the given ``input_id``.

        Returns ``ui.span()`` when choices are unavailable.

        :param input_id: Shiny input ID for this selectize instance.

        """
        choices = _reg_selector_choices()
        if choices is None:
            return ui.span()
        with reactive.isolate():
            cur = selected_reg()
        default = cur if cur in choices else next(iter(choices), "")
        if not default:
            return ui.span()
        return ui.input_selectize(
            input_id, "Regulator", choices=choices, selected=default
        )

    @output(suspend_when_hidden=False)
    @render.ui
    def regulator_selector_box() -> ui.Tag:
        """
        Regulator dropdown on the Pair Distribution tab.

        :trigger _run_analysis.status: re-renders when the task completes. :trigger
        selected_reg: re-renders to reflect the current selection.

        """
        return _reg_selector_tag("selected_regulator_box")

    @output(suspend_when_hidden=False)
    @render.ui
    def regulator_selector_scatter() -> ui.Tag:
        """
        Regulator dropdown on the Gene Scatter tab.

        :trigger _run_analysis.status: re-renders when the task completes. :trigger
        selected_reg: re-renders to reflect the current selection.

        """
        return _reg_selector_tag("selected_regulator_scatter")

    @reactive.effect
    @reactive.event(input.selected_regulator_box)
    def _sync_box_dropdown() -> None:
        """
        Propagate the Pair Distribution dropdown selection to ``selected_reg``.

        :trigger input.selected_regulator_box: fires when the user picks a regulator.

        """
        try:
            val = str(input.selected_regulator_box())
        except Exception:
            return
        with reactive.isolate():
            if val != selected_reg():
                selected_reg.set(val)

    @reactive.effect
    @reactive.event(input.selected_regulator_scatter)
    def _sync_scatter_dropdown() -> None:
        """
        Propagate the Gene Scatter dropdown selection to ``selected_reg``.

        :trigger input.selected_regulator_scatter: fires when the user picks a
        regulator.

        """
        try:
            val = str(input.selected_regulator_scatter())
        except Exception:
            return
        with reactive.isolate():
            if val != selected_reg():
                selected_reg.set(val)

    @reactive.effect
    def _sync_reg_to_dropdowns() -> None:
        """
        Push ``selected_reg`` changes back into both selectize inputs so they stay in
        sync regardless of which one (or a box-plot click) triggered the change.

        :trigger selected_reg: fires whenever the selected regulator changes.

        """
        reg = selected_reg()
        if not reg:
            return
        ui.update_selectize("selected_regulator_box", selected=reg, session=session)
        ui.update_selectize("selected_regulator_scatter", selected=reg, session=session)

    # --- Scatter tab status and plots ------------------------------------------

    @output(suspend_when_hidden=False)
    @render.ui
    def scatter_status() -> ui.Tag:
        """
        Status message on the Gene Scatter tab while plots are being built.

        :trigger _scatter_rendered: re-renders as each scatter plot finishes. :trigger
        _scatter_expected: re-renders when expected count is set.

        """
        if _run_analysis.status() != "success":
            return ui.span()
        if _scatter_rendered() < _scatter_expected():
            return ui.div(
                {"class": "empty-state"},
                ui.p("Preparing visualizations, please wait..."),
            )
        return ui.span()

    @output(suspend_when_hidden=False)
    @render.ui
    def scatter_container() -> ui.Tag:
        """
        Flex container with one slot per active pair that shares a dataset with the
        currently selected pair.  When no pair is selected all active pairs are shown.

        :trigger _run_analysis.status: re-renders when the task completes. :trigger
        selected_pair: re-renders when the selected pair changes.

        """
        # Unmount the scatter figures when the Binding tab is not active so they
        # do not stay resident in the DOM (see pair_box_container).
        if active_tab is not None and active_tab() != "Binding":
            return ui.span()
        if _run_analysis.status() != "success":
            return ui.span()

        result = _run_analysis.result()
        pairs: list[tuple[str, str]] = result["pairs"]
        if not pairs:
            return ui.span()

        # Filter to pairs where both datasets appear in the committed selection.
        # With no selection the scatter stays empty (the matrix is the single
        # source of truth) rather than defaulting to every active pair.
        sel = committed_pairs()
        if not sel:
            return ui.div(
                {"class": "empty-state"},
                ui.p(
                    "Click a cell in the Correlation Matrix and run Execute "
                    "Analysis to view gene-level scatter plots."
                ),
            )
        sel_dbs: set[str] = {db for p in sel for db in p}
        visible = [p for p in pairs if set(p) <= sel_dbs]

        if not visible:
            return ui.span()

        visible_set = set(visible)
        slots = [
            ui.output_ui(f"scatter_{db_a}__{db_b}")
            for db_a, db_b in _all_possible_pairs
            if (db_a, db_b) in visible_set
        ]
        return ui.div(
            ui.output_ui("scatter_missing_note"),
            ui.div(
                *slots,
                style=(
                    "display: flex; flex-wrap: wrap;"
                    " gap: 1rem; align-items: flex-start;"
                ),
            ),
        )

    @output(suspend_when_hidden=False)
    @render.ui
    def scatter_missing_note() -> ui.Tag:
        """
        Warning listing datasets where the selected regulator was not found, scoped to
        the visible pairs only.

        :trigger selected_reg: re-renders when the regulator changes. :trigger
        _run_analysis.status: re-renders when the task completes. :trigger
        committed_pairs: re-renders when the pair selection changes.

        """
        if _run_analysis.status() != "success":
            return ui.span()
        reg = selected_reg()
        if not reg:
            return ui.span()

        result = _run_analysis.result()
        pairs: list[tuple[str, str]] = result["pairs"]
        corr_data = result["corr_data"]

        sel = committed_pairs()
        if sel:
            sel_dbs: set[str] = {db for p in sel for db in p}
            visible = [p for p in pairs if set(p) <= sel_dbs]
        else:
            visible = pairs

        failed: set[str] = set()
        succeeded: set[str] = set()
        for db_a, db_b in visible:
            df = corr_data.get((db_a, db_b))
            has_reg = (
                df is not None
                and not df.empty
                and reg in df["regulator_locus_tag"].values
            )
            if has_reg:
                succeeded.add(display_names.get(db_a, db_a))
                succeeded.add(display_names.get(db_b, db_b))
            else:
                failed.add(display_names.get(db_a, db_a))
                failed.add(display_names.get(db_b, db_b))

        truly_missing = failed - succeeded
        if not truly_missing:
            return ui.span()
        names = ", ".join(sorted(truly_missing))
        return ui.p(
            f"{reg} was not found in: {names}. "
            "Pairs involving these datasets are omitted.",
            style="color: gray; margin: 0.5rem 0;",
        )

    def _make_scatter_render(db_a: str, db_b: str, pair_idx: int) -> None:
        """
        Register a ``@render.ui`` for the scatter plot of one dataset pair.

        Reads params from the task result rather than current sidebar inputs so the plot
        reflects the state at the time Execute was clicked.

        :param db_a: First dataset name.
        :param db_b: Second dataset name.
        :param pair_idx: Stable integer index used to namespace SQL parameters.

        """

        @output(id=f"scatter_{db_a}__{db_b}")
        @render.ui
        async def _scatter_plot() -> ui.Tag:
            """
            Scatter plot for one (db_a, db_b) pair.

            Captures ``_scatter_epoch`` at render-start so that completions from
            a stale epoch (e.g. a previous regulator selection still in flight)
            do not advance the current epoch's counter.

            :trigger selected_reg: re-renders when the regulator changes.
            :trigger _run_analysis.status: re-renders when the task completes.

            """
            with perf(session.id, "binding.workspace", f"scatter_{db_a}__{db_b}"):
                # Gate vdb access until background materialization finishes (also
                # guaranteed transitively via _run_analysis below, but explicit here).
                if materialize_ready is not None:
                    req(materialize_ready())
                # Capture epoch without creating a reactive dep on _scatter_epoch.
                with reactive.isolate():
                    my_epoch = _scatter_epoch()

                def _count_scatter() -> None:
                    """Increment the scatter counter if this render's epoch matches."""
                    with reactive.isolate():
                        if _scatter_epoch() == my_epoch:
                            _scatter_rendered.set(_scatter_rendered() + 1)

                if _run_analysis.status() != "success":
                    return ui.span()

                result = _run_analysis.result()
                pairs: list[tuple[str, str]] = result["pairs"]
                if (db_a, db_b) not in pairs:
                    return ui.span()

                reg = selected_reg()
                if not reg:
                    _count_scatter()
                    return ui.span()

                col_map = result["col_map"]
                method = result["method"]
                filters = result["filters"]

                def _strip_reg(f: dict | None) -> dict | None:
                    if not f:
                        return f
                    stripped = {
                        k: v for k, v in f.items() if k != "regulator_locus_tag"
                    }
                    return stripped or None

                try:
                    col_a = col_map.get(db_a) or get_measurement_column(db_a, "effect")
                    col_b = col_map.get(db_b) or get_measurement_column(db_b, "effect")
                    fa = _strip_reg(filters.get(db_a))
                    fb = _strip_reg(filters.get(db_b))
                    scatter_sql, scatter_params = regulator_scatter_sql(
                        db_a, col_a, fa, db_b, col_b, fb, method, reg, pair_idx
                    )
                    merged = await asyncio.to_thread(
                        vdb.query, scatter_sql, **scatter_params
                    )
                    logger.debug(
                        "binding scatter %s/%s reg=%r rows=%d",
                        db_a,
                        db_b,
                        reg,
                        len(merged),
                    )
                except Exception:
                    logger.exception("Scatter fetch failed for %s/%s", db_a, db_b)
                    _count_scatter()
                    return ui.span()

                if merged.empty:
                    _count_scatter()
                    return ui.span()

                col_preference = result.get("col_preference", "effect")
                la = display_names.get(db_a, db_a)
                lb = display_names.get(db_b, db_b)

                val_a = merged["_val_a"].copy()
                val_b = merged["_val_b"].copy()
                if col_preference == "log10pval" and method != "spearman":
                    val_a, axis_label_a = _apply_log10p_transform(
                        val_a, db_a, col_a, la
                    )
                    val_b, axis_label_b = _apply_log10p_transform(
                        val_b, db_b, col_b, lb
                    )
                elif col_preference == "log10pval" and method == "spearman":
                    # Spearman returns ranks; -log10 transform does not apply.
                    axis_label_a = f"{la}: rank by p-value"
                    axis_label_b = f"{lb}: rank by p-value"
                else:
                    axis_label_a = f"{la}: {col_a}"
                    axis_label_b = f"{lb}: {col_b}"

                r = val_a.corr(val_b)

                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=val_a,
                        y=val_b,
                        mode="markers",
                        marker=dict(size=4, opacity=0.6, color="#4A90D9"),
                        text=merged["target_symbol"],
                        hovertemplate="%{text}<extra></extra>",
                        showlegend=False,
                    )
                )
                fig.add_annotation(
                    text=f"r={r:.3f}",
                    xref="paper",
                    yref="paper",
                    x=0.98,
                    y=0.98,
                    xanchor="right",
                    yanchor="top",
                    showarrow=False,
                    font=dict(size=12),
                )
                fig.update_layout(
                    title=dict(text=f"{la}<br>vs<br>{lb}", x=0.5, xanchor="center"),
                    xaxis_title=axis_label_a,
                    yaxis_title=axis_label_b,
                    margin=dict(l=50, r=20, t=100, b=50),
                    width=400,
                    height=400,
                )
                _count_scatter()
                return ui.div(
                    ui.HTML(to_html(fig, include_plotlyjs=False, full_html=False)),
                    style="flex: 0 0 auto;",
                )

    for _pair_idx, (_db_a, _db_b) in enumerate(_all_possible_pairs, start=1):
        _make_scatter_render(_db_a, _db_b, _pair_idx)


__all__ = ["binding_workspace_server"]
