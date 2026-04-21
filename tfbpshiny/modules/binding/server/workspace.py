from __future__ import annotations

import itertools
from collections.abc import Callable
from html import escape
from logging import Logger
from typing import Any, Literal

import pandas as pd
import plotly.graph_objects as go
from labretriever import VirtualDB
from plotly.io import to_html
from shiny import module, reactive, render, ui

from tfbpshiny.modules.binding.queries import (
    corr_pair_sql,
    get_measurement_column,
    regulator_scatter_sql,
)
from tfbpshiny.utils.profiler import profile_span
from tfbpshiny.utils.sample_conditions import fetch_sample_condition_map
from tfbpshiny.utils.vdb_init import AppDatasets, get_regulator_display_name


@module.server
def binding_workspace_server(
    input: Any,
    output: Any,
    session: Any,
    active_binding_datasets: reactive.Calc_[list[str]],
    corr_type: Callable[[], str],
    col_preference: Callable[[], str],
    dataset_filters: reactive.Value[dict[str, Any]],
    vdb: VirtualDB,
    app_datasets: AppDatasets,
    logger: Logger,
    profile_logger: Logger,
    session_id: str = "",
) -> None:
    """
    Render the binding correlation rows: pairwise distributions
    and per-regulator plots.
    """

    display_names: dict[str, str] = {
        db_name: vdb.get_tags(db_name).get("display_name", db_name)
        for db_name in vdb.get_datasets()
    }

    _reg_df = get_regulator_display_name(vdb)
    sym_map: dict[str, str] = dict(
        zip(_reg_df["regulator_locus_tag"], _reg_df["display_name"])
    )

    @reactive.calc
    def _condition_maps() -> dict[str, dict[str, str]]:
        """
        ``{db_name: {sample_id: label}}`` for each active dataset that has
        experimental_condition columns.

        Used to annotate tooltips on the selected-regulator overlay in the
        distribution plot so the user can distinguish multiple samples of the
        same regulator. Datasets without ``condition_cols`` are omitted from
        the outer dict, causing the tooltip to skip their side.

        :trigger active_binding_datasets: re-runs when the user toggles a
            binding dataset on or off.
        :returns: Outer dict keyed by db_name; inner dict maps sample_id to
            the joined condition label.

        """
        out: dict[str, dict[str, str]] = {}
        for db in active_binding_datasets():
            cols = app_datasets.condition_cols.get(db, [])
            if not cols:
                continue
            try:
                out[db] = fetch_sample_condition_map(vdb, db, cols)
            except Exception:
                logger.exception("Failed to fetch condition map for %s", db)
                out[db] = {}
        return out

    @reactive.calc
    def _pairs() -> list[tuple[str, str]]:
        """
        All unique pairs of active binding datasets.

        :trigger active_binding_datasets: re-runs whenever the user toggles a
            binding dataset on or off in the Select Datasets sidebar.
        :returns: List of ``(db_a, db_b)`` tuples, length = n_active choose 2.

        """
        active = active_binding_datasets()
        pairs = list(itertools.combinations(active, 2))
        logger.debug(f"binding _pairs: active={active}, pairs={pairs}")
        return pairs

    @reactive.calc
    def _all_corr_data() -> dict[tuple[str, str], pd.DataFrame]:
        """
        Per-regulator correlation values for every active dataset pair.

        The heart of this function is a for loop over the dataset pairs. In each
        iteration, the user selected measurement column (effect or p-value),
        filters for that dataset, and the correlation method (pearson or spearman) are
        submitted along with the virtualDB instance to `corr_pair_sql()`
        (see queries.py), which uses the duckDB aggregate functions to compute
        correlations. The join is done on (regulator_locus_tag, target_locus_tag).
        NOTE: if there are multiple samples for a given regulator_locus_tag, then
        there will be multiple correlation values for that regulator in the output
        dataframe.

        :trigger _pairs: re-runs when the set of active pairs changes.
        :trigger col_preference: re-runs when the user switches between Effect
            and P-value columns.
        :trigger corr_type: re-runs when the user switches between Pearson and
            Spearman.
        :trigger dataset_filters: re-runs when filters are applied or reset for
            any dataset.
        :returns: a dict with keys ``(db_a, db_b)`` and values a dataframe
            with columns ``db_a``, ``db_a_id``, ``db_b``, ``db_b_id``,
            ``regulator_locus_tag`` and ``correlation``. Failed pairs are stored as
            empty DataFrames so downstream renders can handle them gracefully. The
            failure and error are logged at the ERROR level.

        """
        pairs = _pairs()
        # TODO: get rid of the type ignore
        preference: Literal["effect", "pvalue"] = col_preference()  # type: ignore[assignment] # noqa: E501
        method = corr_type()
        filters = dataset_filters()

        if not pairs:
            return {}

        result: dict[tuple[str, str], pd.DataFrame] = {}
        for i, (db_a, db_b) in enumerate(pairs):
            try:
                col_a = get_measurement_column(db_a, preference)
                col_b = get_measurement_column(db_b, preference)
                logger.debug(
                    f"Correlating {db_a}({col_a}) vs {db_b}({col_b}) ({method})"
                )
                with profile_span(
                    profile_logger,
                    "vdb.query",
                    module="binding",
                    dataset=f"{db_a}x{db_b}",
                    context="_all_corr_data",
                    session_id=session_id,
                ):
                    result[(db_a, db_b)] = corr_pair_sql(
                        vdb,
                        db_a,
                        col_a,
                        filters.get(db_a),
                        db_b,
                        col_b,
                        filters.get(db_b),
                        method,
                        prefix=f"p{i}_",
                    )
            except Exception as exc:
                logger.error(
                    f"Failed to correlate {db_a} vs {db_b}: {exc}", exc_info=True
                )
                result[(db_a, db_b)] = pd.DataFrame(
                    columns=[
                        "db_a",
                        "db_a_id",
                        "db_b",
                        "db_b_id",
                        "regulator_locus_tag",
                        "correlation",
                    ]
                )

        return result

    @render.ui
    def distributions_plot() -> ui.Tag:
        """
        Box-plot of per-regulator correlation values for every active pair.

        :trigger _pairs: re-renders when the set of active pairs changes. :trigger
        _all_corr_data: re-renders when correlation data is recomputed. :trigger
        corr_type: re-renders when the correlation method changes     (updates the
        axis/title label).

        """
        pairs = _pairs()
        corr_data = _all_corr_data()
        method = corr_type().capitalize()

        fig = go.Figure()

        if not pairs:
            fig.add_annotation(
                text="Select at least two binding datasets to see correlations.",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
            )
            return fig

        # sym_map is built at server init from the pre-computed lookup table
        try:
            selected_reg = str(input.selected_regulator())
        except Exception:
            selected_reg = ""

        cond_maps = _condition_maps()

        # Build a single combined box trace using x as the category axis.
        # Each point's x value is the pair label; Plotly groups points under
        # each category and draws one box per unique x value.
        all_x: list[str] = []
        all_y: list[float] = []
        all_tags: list[str] = []
        all_hover: list[str] = []
        sel_x: list[str] = []
        sel_y: list[float] = []
        sel_hover: list[str] = []
        sel_tags: list[str] = []
        for db_a, db_b in pairs:
            df = corr_data.get((db_a, db_b), pd.DataFrame())
            label_a = display_names.get(db_a, db_a)
            label_b = display_names.get(db_b, db_b)
            pair_label = f"{label_a}<br>vs<br>{label_b}"
            cond_a = cond_maps.get(db_a, {})
            cond_b = cond_maps.get(db_b, {})
            if not df.empty:
                df_clean = df.dropna(subset=["correlation"])
                for tag, corr, sample_a, sample_b in zip(
                    df_clean["regulator_locus_tag"],
                    df_clean["correlation"],
                    df_clean["db_a_id"],
                    df_clean["db_b_id"],
                ):
                    display = sym_map.get(tag, tag)
                    all_x.append(pair_label)
                    all_y.append(corr)
                    all_tags.append(tag)
                    all_hover.append(display)
                    if tag == selected_reg:
                        sel_x.append(pair_label)
                        sel_y.append(corr)
                        # Per-dot hover: regulator + r + one condition line per
                        # dataset that has a non-empty label for this sample.
                        # All DB-sourced strings are HTML-escaped before being
                        # joined with the <br> separators because Plotly renders
                        # hovertext as HTML (stored-XSS sink if any researcher-
                        # uploaded metadata ever contained markup).
                        hover_lines = [escape(display), f"r = {corr:.3f}"]
                        label_sample_a = cond_a.get(str(sample_a), "")
                        if label_sample_a:
                            hover_lines.append(
                                f"{escape(label_a)}: {escape(label_sample_a)}"
                            )
                        label_sample_b = cond_b.get(str(sample_b), "")
                        if label_sample_b:
                            hover_lines.append(
                                f"{escape(label_b)}: {escape(label_sample_b)}"
                            )
                        sel_hover.append("<br>".join(hover_lines))
                        sel_tags.append(tag)

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

        if sel_x:
            fig.add_trace(
                go.Scatter(
                    x=sel_x,
                    y=sel_y,
                    mode="markers",
                    hovertext=sel_hover,
                    customdata=sel_tags,
                    hovertemplate="%{hovertext}<extra></extra>",
                    marker=dict(size=10, color="black", symbol="circle"),
                    showlegend=False,
                )
            )

        fig.update_layout(
            title=f"{method} correlation across regulators",
            yaxis_title=f"{method} r",
            showlegend=False,
            margin=dict(l=40, r=20, t=50, b=80),
        )
        input_id = session.ns("selected_regulator")
        post_script = (
            "(function() {"
            "  var div = document.getElementById('{plot_id}');"
            "  div.on('plotly_click', function(data) {"
            "    var pt = data.points[0];"
            "    if (!pt || pt.customdata === undefined) return;"
            f"    Shiny.setInputValue('{input_id}', pt.customdata, {{priority: 'event'}});"  # type: ignore # noqa:E501
            "  });"
            "})();"
        )
        with profile_span(
            profile_logger,
            "plot.build",
            module="binding",
            context="distributions_plot",
            session_id=session_id,
        ):
            html = to_html(
                fig, include_plotlyjs="cdn", full_html=False, post_script=post_script
            )
        return ui.HTML(html)

    @render.ui
    def regulator_selector() -> ui.Tag:
        """
        Dropdown of regulators present in at least one pair's correlation data.

        Choices are keyed by locus tag and labelled by gene symbol where available. The
        previously selected regulator is preserved across re-renders if it is still
        present in the new choice set.

        :trigger _all_corr_data: re-renders when correlation data changes (new
        datasets selected, filters applied, or column/method changed). :trigger
        active_binding_datasets: re-renders to refresh the symbol map     when the
        active dataset set changes.

        """
        corr_data = _all_corr_data()
        if not corr_data:
            return ui.span()

        # sym_map is built at server init from the pre-computed lookup table
        all_regs: set[str] = set()
        for df in corr_data.values():
            if not df.empty:
                all_regs |= set(df["regulator_locus_tag"].dropna().unique())

        if not all_regs:
            return ui.span()
        choices = {r: sym_map.get(r, r) for r in all_regs}
        choices = dict(sorted(choices.items(), key=lambda kv: kv[1].lower()))

        try:
            current = str(input.selected_regulator())
        except Exception:
            current = ""
        default = current if current in choices else next(iter(choices))

        return ui.input_selectize(
            "selected_regulator",
            "Regulator",
            choices=choices,
            selected=default,
        )

    @render.ui
    def regulator_plots() -> ui.Tag:
        """
        Per-pair scatter plots for the selected regulator.

        Delegates to ``_build_regulator_plots``; catches and renders any
        unhandled exceptions as an annotated empty figure.

        :trigger input.selected_regulator: re-renders when the user picks a
            different regulator from the dropdown.
        :trigger _all_corr_data: re-renders when correlation data changes
            (new datasets, filters, column preference, or method).

        """
        try:
            return _build_regulator_plots()
        except Exception as exc:
            logger.exception("regulator_plots render failed")
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error rendering plots: {exc}",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
            )
            return ui.HTML(to_html(fig, include_plotlyjs=False, full_html=False))

    def _build_regulator_plots() -> ui.Tag:
        try:
            reg = str(input.selected_regulator()) or None
        except Exception:
            reg = None
        pairs = list(_all_corr_data().keys())
        # TODO: get rid of the type ignore
        preference: Literal["effect", "pvalue"] = col_preference()  # type: ignore[assignment] # noqa: E501
        filters = dataset_filters()
        method = corr_type()

        if not reg or not pairs:
            fig = go.Figure()
            fig.add_annotation(
                text="Select a regulator above to see per-pair scatter plots.",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
            )
            return ui.HTML(to_html(fig, include_plotlyjs=False, full_html=False))

        # First pass: collect data and track which datasets are missing the regulator.
        # A dataset is only reported missing if it has no successful pair — a dataset
        # involved in a failed pair but succeeding in another is not "missing".
        pair_data: list[tuple[str, str, str, str, object]] = []
        failed_datasets: set[str] = set()
        succeeded_datasets: set[str] = set()

        def _strip_reg(f: dict | None) -> dict | None:
            # Strip regulator_locus_tag from filters — the scatter query adds its
            # own per-regulator WHERE clause; keeping it would create an AND conflict.
            if not f:
                return f
            stripped = {k: v for k, v in f.items() if k != "regulator_locus_tag"}
            return stripped or None

        for idx, (db_a, db_b) in enumerate(pairs, start=1):
            try:
                col_a = get_measurement_column(db_a, preference)
                col_b = get_measurement_column(db_b, preference)
                fa = _strip_reg(filters.get(db_a))
                fb = _strip_reg(filters.get(db_b))
                scatter_sql, scatter_params = regulator_scatter_sql(
                    db_a,
                    col_a,
                    fa,
                    db_b,
                    col_b,
                    fb,
                    method,
                    reg,
                    idx,
                )
                with profile_span(
                    profile_logger,
                    "vdb.query",
                    module="binding",
                    context="scatter",
                    session_id=session_id,
                ):
                    merged = vdb.query(scatter_sql, **scatter_params)
                logger.debug(
                    f"scatter {db_a}/{db_b} reg={reg!r} "
                    f"rows={len(merged)} fa={fa!r} fb={fb!r}"
                )
            except Exception:
                logger.exception(f"Regulator plot fetch failed for {db_a}/{db_b}")
                continue

            if merged.empty:
                failed_datasets.add(display_names.get(db_a, db_a))
                failed_datasets.add(display_names.get(db_b, db_b))
                continue

            succeeded_datasets.add(display_names.get(db_a, db_a))
            succeeded_datasets.add(display_names.get(db_b, db_b))
            pair_data.append((db_a, db_b, col_a, col_b, merged))

        # Build one figure per valid pair
        plot_divs: list[ui.Tag] = []
        for db_a, db_b, col_a, col_b, merged in pair_data:
            la = display_names.get(db_a, db_a)
            lb = display_names.get(db_b, db_b)
            r = merged["_val_a"].corr(merged["_val_b"])

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=merged["_val_a"],
                    y=merged["_val_b"],
                    mode="markers",
                    marker=dict(size=4, opacity=0.6, color="#4A90D9"),
                    text=merged["target_locus_tag"],
                    hovertemplate=(
                        f"{la}: %{{x:.3f}}<br>" + f"{lb}: %{{y:.3f}}<extra></extra>"
                    ),
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
                title=dict(
                    text=f"{la}<br>vs<br>{lb}",
                    x=0.5,
                    xanchor="center",
                ),
                xaxis_title=f"{la}: {col_a}",
                yaxis_title=f"{lb}: {col_b}",
                margin=dict(l=50, r=20, t=100, b=50),
                width=400,
                height=400,
            )
            with profile_span(
                profile_logger,
                "plot.build",
                module="binding",
                context="regulator_plot",
                session_id=session_id,
            ):
                plot_html = to_html(fig, include_plotlyjs=False, full_html=False)
            plot_divs.append(
                ui.div(
                    ui.HTML(plot_html),
                    style="flex: 0 0 auto;",
                )
            )

        missing_note: list[ui.Tag] = []
        truly_missing = failed_datasets - succeeded_datasets
        if truly_missing:
            names = ", ".join(sorted(truly_missing))
            missing_note.append(
                ui.p(
                    f"{reg} was not found in: {names}. "
                    "Pairs involving these datasets are omitted.",
                    style="color: gray; margin: 0.5rem 0;",
                )
            )

        return ui.div(
            *missing_note,
            ui.div(
                *plot_divs,
                style="display: flex; flex-wrap: wrap; gap: 1rem; align-items: flex-start;",  # noqa: E501
            ),
        )


__all__ = ["binding_workspace_server"]
