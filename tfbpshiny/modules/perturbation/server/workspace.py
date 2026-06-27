from __future__ import annotations

import itertools
from logging import Logger
from typing import Any

import duckdb
import pandas as pd
import plotly.graph_objects as go
from plotly.io import to_html
from shiny import module, reactive, render, ui

from tfbpshiny.modules.perturbation.queries import fetch_corr_pairs
from tfbpshiny.utils.correlation_matrix import build_correlation_matrix_ui
from tfbpshiny.utils.vdb_init import get_regulator_display_name


def _read_corr_type(input: Any) -> str:
    try:
        return str(input.corr_type())
    except Exception:
        return "pearson"


@module.server
def perturbation_workspace_server(
    input: Any,
    output: Any,
    session: Any,
    active_perturbation_datasets: reactive.Calc_[list[str]],
    dataset_filters: reactive.Value[dict[str, Any]],
    conn: duckdb.DuckDBPyConnection,
    logger: Logger,
) -> None:
    """
    Render the perturbation correlation workspace: matrix, pair distribution box plots.

    :param active_perturbation_datasets: Reactive calc returning active perturbation
        db names.
    :param dataset_filters: Reactive value holding current filter state.
    :param conn: Read-only DuckDB connection to the materialized database.
    :param logger: Application logger.

    """
    _display_df = conn.execute(
        "SELECT db_name, display_name FROM dataset_registry"
    ).df()
    display_names: dict[str, str] = dict(
        zip(_display_df["db_name"], _display_df["display_name"])
    )

    _reg_df = get_regulator_display_name(conn)
    _sym_lookup: dict[str, str] = {}
    for _, row in _reg_df.iterrows():
        tag = str(row["regulator_locus_tag"])
        sym = str(row.get("regulator_symbol", ""))
        if sym and sym != "nan" and sym != tag:
            _sym_lookup[tag] = f"{sym} ({tag})"
        else:
            _sym_lookup[tag] = tag

    _pert_dbs_df = conn.execute(
        "SELECT db_name FROM dataset_registry WHERE data_type = 'perturbation' ORDER BY db_name"
    ).df()
    _all_possible_pairs: list[tuple[str, str]] = list(
        itertools.combinations(_pert_dbs_df["db_name"].tolist(), 2)
    )

    _selected_pairs: reactive.Value[list[tuple[str, str]]] = reactive.value([])
    _execute_trigger: reactive.Value[int] = reactive.value(0)

    @reactive.effect
    @reactive.event(input.execute_analysis)
    def _on_execute() -> None:
        """
        Force a re-run of the correlation data fetch.

        :trigger: ``input.execute_analysis`` — fires when Execute Analysis is clicked.

        """
        _execute_trigger.set(_execute_trigger() + 1)

    @reactive.calc
    def _active_pairs() -> list[tuple[str, str]]:
        """
        Pairs of active perturbation datasets.

        :trigger: ``active_perturbation_datasets`` — re-runs when selection changes.

        """
        active = active_perturbation_datasets()
        return list(itertools.combinations(sorted(active), 2))

    @reactive.calc
    def _corr_data() -> dict[tuple[str, str], pd.DataFrame]:
        """
        Per-regulator correlation values for every active dataset pair.

        :trigger: ``_active_pairs`` — re-runs when pairs change.
        :trigger: ``input.corr_type`` — re-runs when correlation method changes.
        :trigger: ``dataset_filters`` — re-runs when filters change.
        :trigger: ``_execute_trigger`` — re-runs when Execute is clicked.

        """
        _execute_trigger()
        pairs = _active_pairs()
        method = _read_corr_type(input)
        filters = dataset_filters()

        if not pairs:
            return {}

        logger.debug("perturbation _corr_data: pairs=%s method=%s", pairs, method)
        try:
            return fetch_corr_pairs(
                conn, pairs, filters, method, comparison_type="perturbation"
            )
        except Exception:
            logger.exception("perturbation fetch_corr_pairs failed")
            return {p: pd.DataFrame() for p in pairs}

    @render.ui
    def execute_pending_style() -> ui.Tag:
        """No pending concept with pre-materialized data."""
        return ui.span()

    @render.ui
    def hackett_pvalue_warning() -> ui.Tag:
        """Placeholder for any dataset-specific warnings."""
        return ui.span()

    @render.ui
    def analysis_status() -> ui.Tag:
        """
        Status message when no pairs are available.

        :trigger: ``_active_pairs`` — re-renders when pair list changes.

        """
        if not _active_pairs():
            return ui.div(
                {"class": "empty-state"},
                ui.p("Select at least two perturbation datasets to see correlations."),
            )
        return ui.span()

    @render.ui
    def corr_matrix_container() -> ui.Tag:
        """
        Pairwise correlation matrix table.

        :trigger: ``_active_pairs`` — re-renders when pairs change.
        :trigger: ``_corr_data`` — re-renders when data is refreshed.
        :trigger: ``_selected_pairs`` — re-renders to update cell highlights.

        """
        pairs = _active_pairs()
        if not pairs:
            return ui.div(
                {"class": "empty-state"},
                ui.p("Click Execute Analysis after selecting datasets."),
            )
        corr_data = _corr_data()
        active = active_perturbation_datasets()
        return build_correlation_matrix_ui(
            all_possible_pairs=_all_possible_pairs,
            active_pairs=pairs,
            active_datasets=sorted(active),
            corr_data=corr_data,
            display_names=display_names,
            selected_pairs=set(_selected_pairs()),
            ns=session.ns,
        )

    def _make_cell_click_effect(db_a: str, db_b: str) -> None:
        btn_id = f"corrpair_{db_a}__{db_b}"
        pair = (db_a, db_b)

        @reactive.effect
        @reactive.event(input[btn_id])
        def _on_click() -> None:
            cur = list(_selected_pairs())
            if pair in cur:
                cur.remove(pair)
            else:
                cur.append(pair)
            _selected_pairs.set(cur)

    for _db_a, _db_b in _all_possible_pairs:
        _make_cell_click_effect(_db_a, _db_b)

    @reactive.effect
    def _prune_selected_pairs() -> None:
        """
        Remove stale selected pairs when the active pair set changes.

        :trigger: ``_active_pairs`` — fires when datasets change.

        """
        active_set = set(_active_pairs())
        with reactive.isolate():
            cur = _selected_pairs()
        valid = [p for p in cur if p in active_set]
        if valid != cur:
            _selected_pairs.set(valid)

    @render.ui
    def regulator_selector_box() -> ui.Tag:
        """
        Dropdown of regulators for highlighting in the pair distribution plots.

        :trigger: ``_corr_data`` — re-renders when data changes.
        :trigger: ``_selected_pairs`` — re-renders when pair selection changes.

        """
        sel_pairs = _selected_pairs()
        if not sel_pairs:
            return ui.span()
        corr_data = _corr_data()
        all_regs: set[str] = set()
        for pair in sel_pairs:
            df = corr_data.get(pair, pd.DataFrame())
            if not df.empty and "regulator_locus_tag" in df.columns:
                all_regs |= set(df["regulator_locus_tag"].dropna().astype(str))
        if not all_regs:
            return ui.span()
        choices = {r: _sym_lookup.get(r, r) for r in all_regs}
        choices = dict(sorted(choices.items(), key=lambda kv: kv[1].lower()))
        try:
            cur = str(input.selected_reg_box())
        except Exception:
            cur = ""
        default = cur if cur in choices else next(iter(choices))
        return ui.input_selectize(
            "selected_reg_box", "Highlight regulator", choices=choices, selected=default
        )

    @render.ui
    def pair_box_status() -> ui.Tag:
        """
        Prompt shown when no matrix cell has been selected yet.

        :trigger: ``_active_pairs`` — re-renders when pairs change.
        :trigger: ``_selected_pairs`` — re-renders when selection changes.

        """
        if not _active_pairs():
            return ui.span()
        if not _selected_pairs():
            return ui.div(
                {"class": "empty-state"},
                ui.p("Click a cell in the Correlation Matrix to view its distribution."),
            )
        return ui.span()

    @render.ui
    def pair_box_container() -> ui.Tag:
        """
        Box plots for each selected dataset pair.

        :trigger: ``_selected_pairs`` — re-renders when selection changes.
        :trigger: ``_corr_data`` — re-renders when data changes.
        :trigger: ``input.selected_reg_box`` — re-renders to update highlight.
        :trigger: ``input.corr_type`` — re-renders to update axis label.

        """
        sel_pairs = _selected_pairs()
        if not sel_pairs:
            return ui.span()

        corr_data = _corr_data()
        method = _read_corr_type(input).capitalize()
        try:
            selected_reg = str(input.selected_reg_box())
        except Exception:
            selected_reg = ""

        plots: list[ui.Tag] = []
        for db_a, db_b in sel_pairs:
            df = corr_data.get((db_a, db_b), pd.DataFrame())
            label_a = display_names.get(db_a, db_a)
            label_b = display_names.get(db_b, db_b)
            pair_label = f"{label_a} vs {label_b}"

            fig = go.Figure()
            all_x: list[str] = []
            all_y: list[float] = []
            all_tags: list[str] = []
            all_hover: list[str] = []

            if not df.empty and "correlation" in df.columns:
                df_clean = df.dropna(subset=["correlation"])
                for tag, corr in zip(
                    df_clean["regulator_locus_tag"], df_clean["correlation"]
                ):
                    all_x.append(pair_label)
                    all_y.append(float(corr))
                    all_tags.append(str(tag))
                    all_hover.append(_sym_lookup.get(str(tag), str(tag)))

            fig.add_trace(
                go.Box(
                    x=all_x,
                    y=all_y,
                    text=all_hover,
                    customdata=all_tags,
                    hovertemplate="%{text}<br>r = %{y:.3f}<extra></extra>",
                    hoveron="points",
                    boxpoints="all",
                    jitter=0.4,
                    pointpos=0,
                    marker=dict(size=4, opacity=0.5),
                    line=dict(width=1.5),
                    showlegend=False,
                )
            )

            if selected_reg:
                sel_idx = [i for i, t in enumerate(all_tags) if t == selected_reg]
                if sel_idx:
                    fig.add_trace(
                        go.Scatter(
                            x=[all_x[i] for i in sel_idx],
                            y=[all_y[i] for i in sel_idx],
                            mode="markers",
                            text=[all_hover[i] for i in sel_idx],
                            hovertemplate="%{text}<br>r = %{y:.3f}<extra></extra>",
                            marker=dict(size=10, color="black", symbol="circle"),
                            showlegend=False,
                        )
                    )

            fig.update_layout(
                title=pair_label,
                yaxis_title=f"{method} r",
                margin=dict(l=40, r=20, t=50, b=60),
            )
            plots.append(
                ui.div(
                    ui.HTML(to_html(fig, include_plotlyjs="cdn", full_html=False)),
                    style="flex: 0 0 auto; min-width: 400px;",
                )
            )

        if not plots:
            return ui.span()
        return ui.div(
            *plots,
            style="display: flex; flex-wrap: wrap; gap: 1rem; align-items: flex-start;",
        )


__all__ = ["perturbation_workspace_server"]
