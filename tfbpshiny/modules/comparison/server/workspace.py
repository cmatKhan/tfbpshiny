"""Workspace server for the Comparison module — Phase 2 DuckDB version."""

from __future__ import annotations

from logging import Logger
from typing import Any

import duckdb
import pandas as pd
import plotly.graph_objects as go
from plotly.io import to_html
from plotly.subplots import make_subplots
from shiny import module, reactive, render, ui

from tfbpshiny.components import sidebar_label
from tfbpshiny.modules.comparison.queries import (
    BINDING_BASE_LABEL_MAP,
    BINDING_LABEL_MAP,
    DEFAULT_TOP_N,
    METHOD_BASE_LABEL_MAP,
    PEAKS_VARIANT_MAP,
    PERTURBATION_LABEL_MAP,
    PROMOTER_SET_MAP,
    PROMOTER_VARIANT_PAIRS,
    SCORING_VARIANT_COLORS,
    SCORING_VARIANT_MAP,
    SCORING_VARIANT_ORDER,
    fetch_topn_results,
)
from tfbpshiny.utils.topn_matrix import build_topn_matrix_ui
from tfbpshiny.utils.vdb_init import (
    DEFAULT_RESPONSIVENESS_PRESET,
    DEFAULT_RESPONSIVENESS_PRESETS,
    get_regulator_display_name,
    get_responsiveness_label,
)

_PERT_ORDER = [
    "2006 Overexpression",
    "2006 TFKO",
    "2007 TFKO",
    "2014 TFKO",
    "2020 Overexpression",
    "2025 Degron",
]

_BINDING_ORDER = [
    "2004 ChIP-chip",
    "2021 ChIP-exo",
    "2025 ChEC-seq",
    "2026 Calling Cards",
]

_PROMOTER_SET_ALIAS: dict[str, str] = {
    "Kang": "Promoter Set 1 (Kang)",
    "Mindel": "Promoter Set 2 (Mindel)",
    "500bp": "Promoter Set 3 (500bp)",
    "Intergenic": "Promoter Set 4 (Intergenic)",
}

_PROMOTER_TOOLTIPS: dict[str, str] = {
    "Kang": (
        "700 bp upstream of each start codon, truncated if there exists a feature "
        "within 700 bp of the ORF."
    ),
    "Mindel": (
        "Promoter regions defined from the start codon to at least 700 bp upstream "
        "of the TSS defined by Park et al., 2014; Pelechano et al., 2013; Policastro "
        "et al., 2020 (provided in the SGD annotations). If no TSS is defined, the "
        "start codon is used."
    ),
    "500bp": (
        "Promoter regions defined as exactly 500 bp upstream of the start codon. "
        "No truncation or extension; all promoters are the same length."
    ),
    "Intergenic": (
        "Promoter regions defined as the full intergenic region upstream of the 5' "
        "end of each feature. Note that approximately 1410 of 6040 features are "
        "divergently transcribed."
    ),
}


def _read_top_n(input: Any) -> int:
    try:
        return int(input.top_n())
    except Exception:
        return DEFAULT_TOP_N


def _read_preset(input: Any) -> dict[str, tuple[float, float]]:
    try:
        name = str(input.responsiveness_preset())
    except Exception:
        name = DEFAULT_RESPONSIVENESS_PRESET
    return DEFAULT_RESPONSIVENESS_PRESETS.get(
        name, DEFAULT_RESPONSIVENESS_PRESETS[DEFAULT_RESPONSIVENESS_PRESET]
    )


def _read_preset_name(input: Any) -> str:
    try:
        return str(input.responsiveness_preset())
    except Exception:
        return DEFAULT_RESPONSIVENESS_PRESET


def _cell_style(val: float) -> str:
    pct = max(0.0, min(100.0, val))
    g = int(200 - pct * 1.5)
    return (
        f"padding: 6px 10px; text-align: right;"
        f" background-color: rgb({int(pct * 2.0)},{g},200);"
    )


@module.server
def comparison_workspace_server(
    input: Any,
    output: Any,
    session: Any,
    active_binding_datasets: reactive.Calc_[list[str]],
    active_perturbation_datasets: reactive.Calc_[list[str]],
    dataset_filters: reactive.Value[dict[str, Any]],
    conn: duckdb.DuckDBPyConnection,
    logger: Logger,
) -> None:
    """
    Render the Comparison workspace: topN matrix, distributions, promoter/method tables.

    :param active_binding_datasets: Reactive calc returning active primary binding
        db names.
    :param active_perturbation_datasets: Reactive calc returning active perturbation
        db names.
    :param dataset_filters: Reactive value with per-dataset filter specs.
    :param conn: Read-only DuckDB connection to the materialized database.
    :param logger: Application logger.

    """
    # Pre-load display names and regulator labels from DuckDB.
    _display_df = conn.execute(
        "SELECT db_name, display_name FROM dataset_registry"
    ).df()
    display_names: dict[str, str] = dict(
        zip(_display_df["db_name"], _display_df["display_name"])
    )

    _reg_df = get_regulator_display_name(conn)
    _reg_labels: dict[str, str] = {}
    for _, row in _reg_df.iterrows():
        tag = str(row["regulator_locus_tag"])
        sym = str(row.get("regulator_symbol", ""))
        if sym and sym != "nan" and sym != tag:
            _reg_labels[tag] = f"{sym} ({tag})"
        else:
            _reg_labels[tag] = tag

    _all_binding_dbs = conn.execute(
        "SELECT db_name FROM dataset_registry WHERE data_type = 'binding'"
    ).df()["db_name"].tolist()
    _all_perturbation_dbs = conn.execute(
        "SELECT db_name FROM dataset_registry WHERE data_type = 'perturbation'"
    ).df()["db_name"].tolist()

    # ---------------------------------------------------------------------------
    # State
    # ---------------------------------------------------------------------------

    _execute_trigger: reactive.Value[int] = reactive.value(0)
    _results_stale: reactive.Value[bool] = reactive.value(True)
    cd_selected_binding: reactive.Value[str | None] = reactive.value(None)
    cd_selected_perturbation: reactive.Value[str | None] = reactive.value(None)

    @reactive.effect
    @reactive.event(input.execute_analysis)
    def _on_execute() -> None:
        """
        Trigger a fresh data fetch and mark results as available.

        :trigger: ``input.execute_analysis`` — fires when Execute Analysis is clicked.

        """
        _execute_trigger.set(_execute_trigger() + 1)
        _results_stale.set(False)

    @reactive.effect
    def _mark_stale_on_filter_change() -> None:
        """
        Mark results stale when dataset filters change (Apply Changes was pressed).

        :trigger: ``dataset_filters`` — fires when filters are committed.

        """
        dataset_filters()
        with reactive.isolate():
            if not _results_stale():
                _results_stale.set(True)

    # ---------------------------------------------------------------------------
    # Inner tab helper
    # ---------------------------------------------------------------------------

    def _inner_tab() -> str:
        try:
            return str(input.comparison_inner_tabs())
        except Exception:
            return "Compare Datasets"

    # ---------------------------------------------------------------------------
    # Binding db resolution helpers
    # ---------------------------------------------------------------------------

    def _cd_binding_dbs() -> list[str]:
        """
        Resolve binding db_names for the Compare Datasets tab.

        Maps primary binding datasets to variant db_names based on the selected
        Binding Method and Promoter Set controls.
        """
        try:
            method = str(input.cd_binding_method())
        except Exception:
            method = "Promoter Enrichment"
        try:
            ps = str(input.cd_promoter_set())
        except Exception:
            ps = "Kang"

        active = active_binding_datasets()
        if method == "Peaks":
            result = []
            for b_db in active:
                result.extend(PEAKS_VARIANT_MAP.get(b_db, []))
            return result

        if ps == "Kang":
            return list(active)
        suffix = {
            "Mindel": "_mindel",
            "500bp": "_500bp",
            "Intergenic": "_intergenic",
        }.get(ps, "")
        return [b_db + suffix for b_db in active]

    def _cp_binding_dbs() -> list[str]:
        """
        All binding db_names (primary + variants) for the Compare Promoter tab.

        Builds the set of all active primary datasets plus all their
        promoter-set variant db_names, filtered to only the promoter sets the
        user has checked in the sidebar.
        """
        try:
            included_ps = list(input.cp_included_promoter_sets())
        except Exception:
            included_ps = list(_PROMOTER_SET_ALIAS.keys())

        active = active_binding_datasets()
        result: list[str] = []
        for b_db in active:
            if "Kang" in included_ps:
                result.append(b_db)
            variants = PROMOTER_VARIANT_PAIRS.get(b_db, [])
            for v_db in variants:
                if PROMOTER_SET_MAP.get(v_db, "Kang") in included_ps:
                    result.append(v_db)
        return result

    def _cm_binding_dbs() -> list[str]:
        """
        Binding db_names for the Compare Methods tab (all variants of one dataset).

        Builds enrichment variants (Kang + selected promoter sets) plus the
        peaks variant if available.
        """
        try:
            cm_binding_db = str(input.cm_binding_dataset())
        except Exception:
            cm_binding_db = ""
        if not cm_binding_db:
            active = active_binding_datasets()
            eligible = [db for db in active if db in PEAKS_VARIANT_MAP]
            cm_binding_db = eligible[0] if eligible else (active[0] if active else "")
        if not cm_binding_db:
            return []

        try:
            cm_ps = list(input.cm_promoter_set())
        except Exception:
            cm_ps = ["Kang"]

        result: list[str] = []
        if "Kang" in cm_ps:
            result.append(cm_binding_db)
        for v_db in PROMOTER_VARIANT_PAIRS.get(cm_binding_db, []):
            if PROMOTER_SET_MAP.get(v_db, "Kang") in cm_ps:
                result.append(v_db)
        peaks = PEAKS_VARIANT_MAP.get(cm_binding_db, [])
        result.extend(peaks)
        return result

    # ---------------------------------------------------------------------------
    # Data fetches (triggered by Execute)
    # ---------------------------------------------------------------------------

    @reactive.calc
    def _cd_data() -> pd.DataFrame:
        """
        TopN data for the Compare Datasets tab.

        :trigger: ``_execute_trigger`` — re-runs when Execute is clicked.
        :trigger: ``active_binding_datasets`` — re-runs when binding selection changes.
        :trigger: ``active_perturbation_datasets`` — re-runs when perturbation changes.
        :trigger: ``dataset_filters`` — re-runs when filters change.

        """
        _execute_trigger()
        if _results_stale():
            return pd.DataFrame()
        b_dbs = _cd_binding_dbs()
        p_dbs = active_perturbation_datasets()
        if not b_dbs or not p_dbs:
            return pd.DataFrame()
        pairs = [(b, p) for b in b_dbs for p in p_dbs]
        filters = dataset_filters()
        n = _read_top_n(input)
        preset = _read_preset(input)
        try:
            raw = fetch_topn_results(conn, pairs, filters, n, preset)
        except Exception:
            logger.exception("cd_data fetch failed")
            return pd.DataFrame()
        if raw.empty:
            return pd.DataFrame()
        raw["binding_db"] = raw["pair_key"].str.split("__").str[0]
        raw["perturbation_db"] = raw["pair_key"].str.split("__").str[1]
        raw["binding_label"] = raw["binding_db"].map(
            lambda k: BINDING_LABEL_MAP.get(k, k)
        )
        raw["perturbation_source"] = raw["perturbation_db"].map(
            lambda k: PERTURBATION_LABEL_MAP.get(k, k)
        )
        raw["regulator_label"] = (
            raw["regulator_locus_tag"].map(_reg_labels).fillna(raw["regulator_locus_tag"])
        )
        raw["percent_responsive"] = raw["responsive_ratio"] * 100
        return raw

    @reactive.calc
    def _cp_data() -> pd.DataFrame:
        """
        TopN data for the Compare Promoter Definitions tab.

        :trigger: ``_execute_trigger`` — re-runs when Execute is clicked.

        """
        _execute_trigger()
        if _results_stale():
            return pd.DataFrame()
        b_dbs = _cp_binding_dbs()
        p_dbs = active_perturbation_datasets()
        if not b_dbs or not p_dbs:
            return pd.DataFrame()
        pairs = [(b, p) for b in b_dbs for p in p_dbs]
        filters = dataset_filters()
        n = _read_top_n(input)
        preset = _read_preset(input)
        try:
            raw = fetch_topn_results(conn, pairs, filters, n, preset)
        except Exception:
            logger.exception("cp_data fetch failed")
            return pd.DataFrame()
        if raw.empty:
            return pd.DataFrame()
        raw["binding_db"] = raw["pair_key"].str.split("__").str[0]
        raw["perturbation_db"] = raw["pair_key"].str.split("__").str[1]
        raw["binding_base_label"] = raw["binding_db"].map(
            lambda k: BINDING_BASE_LABEL_MAP.get(k, k)
        )
        raw["promoter_set"] = raw["binding_db"].map(
            lambda k: PROMOTER_SET_MAP.get(k, "Kang")
        )
        raw["perturbation_source"] = raw["perturbation_db"].map(
            lambda k: PERTURBATION_LABEL_MAP.get(k, k)
        )
        raw["regulator_label"] = (
            raw["regulator_locus_tag"].map(_reg_labels).fillna(raw["regulator_locus_tag"])
        )
        raw["percent_responsive"] = raw["responsive_ratio"] * 100
        return raw

    @reactive.calc
    def _cm_data() -> pd.DataFrame:
        """
        TopN data for the Compare Analysis Methods tab.

        :trigger: ``_execute_trigger`` — re-runs when Execute is clicked.

        """
        _execute_trigger()
        if _results_stale():
            return pd.DataFrame()
        b_dbs = _cm_binding_dbs()
        p_dbs = active_perturbation_datasets()
        if not b_dbs or not p_dbs:
            return pd.DataFrame()
        pairs = [(b, p) for b in b_dbs for p in p_dbs]
        filters = dataset_filters()
        n = _read_top_n(input)
        preset = _read_preset(input)
        try:
            raw = fetch_topn_results(conn, pairs, filters, n, preset)
        except Exception:
            logger.exception("cm_data fetch failed")
            return pd.DataFrame()
        if raw.empty:
            return pd.DataFrame()
        raw["binding_db"] = raw["pair_key"].str.split("__").str[0]
        raw["perturbation_db"] = raw["pair_key"].str.split("__").str[1]
        raw["scoring_variant"] = raw["binding_db"].map(
            lambda k: SCORING_VARIANT_MAP.get(k, k)
        )
        raw["perturbation_source"] = raw["perturbation_db"].map(
            lambda k: PERTURBATION_LABEL_MAP.get(k, k)
        )
        raw["regulator_label"] = (
            raw["regulator_locus_tag"].map(_reg_labels).fillna(raw["regulator_locus_tag"])
        )
        raw["percent_responsive"] = raw["responsive_ratio"] * 100
        return raw

    # ---------------------------------------------------------------------------
    # Renders
    # ---------------------------------------------------------------------------

    @render.ui
    def execute_pending_style() -> ui.Tag:
        """Dims the Execute button when no changes are pending."""
        if _results_stale():
            return ui.span()
        return ui.span()

    @render.ui
    def tab_specific_controls() -> ui.Tag:
        """
        Sidebar controls specific to the active inner tab.

        :trigger: ``input.comparison_inner_tabs`` — re-renders when tab changes.

        """
        tab = _inner_tab()
        active = active_binding_datasets()

        if tab == "Compare Datasets":
            return ui.div(
                sidebar_label("Binding Method"),
                ui.input_select(
                    "cd_binding_method",
                    label=None,
                    choices={
                        "Promoter Enrichment": "Promoter Enrichment",
                        "Peaks": "Peaks",
                    },
                    selected="Promoter Enrichment",
                ),
                sidebar_label("Promoter Set"),
                ui.input_select(
                    "cd_promoter_set",
                    label=None,
                    choices={k: _PROMOTER_SET_ALIAS[k] for k in _PROMOTER_SET_ALIAS},
                    selected="Kang",
                ),
            )

        if tab == "Compare Promoter Definitions":
            return ui.div(
                sidebar_label("Promoter Sets"),
                ui.input_checkbox_group(
                    "cp_included_promoter_sets",
                    label=None,
                    choices={
                        ps: ui.tooltip(
                            ui.span(_PROMOTER_SET_ALIAS[ps]),
                            _PROMOTER_TOOLTIPS[ps],
                            placement="right",
                        )
                        for ps in _PROMOTER_SET_ALIAS
                    },
                    selected=list(_PROMOTER_SET_ALIAS.keys()),
                ),
            )

        if tab == "Compare Analysis Methods":
            eligible = [db for db in active if db in PEAKS_VARIANT_MAP]
            method_choices = {db: BINDING_LABEL_MAP.get(db, db) for db in eligible}
            return ui.div(
                sidebar_label("Binding Dataset"),
                ui.input_select(
                    "cm_binding_dataset",
                    label=None,
                    choices=method_choices,
                    selected=next(iter(method_choices), None),
                ),
                sidebar_label("Promoter Set"),
                ui.input_checkbox_group(
                    "cm_promoter_set",
                    label=None,
                    choices={k: _PROMOTER_SET_ALIAS[k] for k in _PROMOTER_SET_ALIAS},
                    selected=["Kang"],
                ),
            )

        return ui.span()

    @render.ui
    def analysis_status() -> ui.Tag:
        """
        Status message when dataset selection is incomplete.

        :trigger: ``active_binding_datasets`` / ``active_perturbation_datasets``.

        """
        if not active_binding_datasets() or not active_perturbation_datasets():
            return ui.div(
                {"class": "empty-state"},
                ui.p("Select at least one binding and one perturbation dataset."),
            )
        return ui.span()

    # ---------------------------------------------------------------------------
    # Tab 1: Compare Datasets — matrix + distribution
    # ---------------------------------------------------------------------------

    def _make_cd_row_effect(b_db: str) -> None:
        @reactive.effect
        @reactive.event(input[f"topnrow_{b_db}"])
        def _on_row() -> None:
            cd_selected_binding.set(b_db)
            cd_selected_perturbation.set(None)

    def _make_cd_col_effect(p_db: str) -> None:
        @reactive.effect
        @reactive.event(input[f"topncol_{p_db}"])
        def _on_col() -> None:
            cd_selected_perturbation.set(p_db)
            cd_selected_binding.set(None)

    for _b in _all_binding_dbs:
        _make_cd_row_effect(_b)
    for _p in _all_perturbation_dbs:
        _make_cd_col_effect(_p)

    @render.ui
    def cd_matrix_container() -> ui.Tag:
        """
        Binding × perturbation top-N responsive ratio matrix.

        :trigger: ``_cd_data`` — re-renders when data changes.
        :trigger: ``cd_selected_binding`` / ``cd_selected_perturbation`` — highlights.

        """
        if _results_stale():
            return ui.div(
                {"class": "empty-state"},
                ui.p("Click Execute Analysis to compute."),
            )
        df = _cd_data()
        b_dbs = _cd_binding_dbs()
        p_dbs = active_perturbation_datasets()
        if not b_dbs or not p_dbs:
            return ui.div(
                {"class": "empty-state"},
                ui.p("No datasets selected."),
            )

        topn_medians: dict[tuple[str, str], float | None] = {}
        if not df.empty:
            for (b_db, p_db), grp in df.groupby(["binding_db", "perturbation_db"]):
                med = grp["percent_responsive"].median()
                topn_medians[(str(b_db), str(p_db))] = (
                    float(med) if pd.notna(med) else None
                )

        preset_name = _read_preset_name(input)

        def _col_tooltip(p_db: str) -> str:
            thresh = get_responsiveness_label(preset_name, p_db)
            return (
                f"Responsive threshold: {thresh}. "
                "Click to view distributions for this perturbation dataset."
            )

        return build_topn_matrix_ui(
            binding_datasets=b_dbs,
            perturbation_datasets=p_dbs,
            topn_medians=topn_medians,
            display_names={**display_names, **BINDING_LABEL_MAP, **PERTURBATION_LABEL_MAP},
            selected_binding=cd_selected_binding(),
            selected_perturbation=cd_selected_perturbation(),
            ns=session.ns,
            col_tooltip=_col_tooltip,
        )

    @render.ui
    def cd_distribution_container() -> ui.Tag:
        """
        Box plots for the selected matrix row or column.

        :trigger: ``_cd_data`` — re-renders when data changes.
        :trigger: ``cd_selected_binding`` / ``cd_selected_perturbation`` — selection.

        """
        if _results_stale():
            return ui.span()
        df = _cd_data()
        if df.empty:
            return ui.span()

        b_sel = cd_selected_binding()
        p_sel = cd_selected_perturbation()

        if b_sel is None and p_sel is None:
            return ui.div(
                {"class": "empty-state"},
                ui.p(
                    "Click a row header to view distributions for a binding dataset,"
                    " or a column header to view distributions for a perturbation"
                    " dataset."
                ),
            )

        if b_sel is not None:
            sub = df[df["binding_db"] == b_sel]
            x_col = "perturbation_source"
        else:
            sub = df[df["perturbation_db"] == p_sel]
            x_col = "binding_label"

        if sub.empty:
            return ui.div(
                {"class": "empty-state"},
                ui.p("No data for the selected datasets."),
            )

        fig = go.Figure()
        x_vals = sorted(sub[x_col].dropna().unique(), key=lambda v: str(v))
        for x_val in x_vals:
            grp = sub[sub[x_col] == x_val]
            mask = grp["percent_responsive"].notna()
            fig.add_trace(
                go.Box(
                    x=grp.loc[mask, x_col].values,
                    y=grp.loc[mask, "percent_responsive"].values,
                    name=str(x_val),
                    text=grp.loc[mask, "regulator_label"].values,
                    hovertemplate="%{text}<br>%{y:.1f}%<extra></extra>",
                    hoveron="points",
                    boxpoints="all",
                    jitter=0.4,
                    pointpos=0,
                    marker=dict(size=4, opacity=0.5),
                    line=dict(width=1.5),
                    showlegend=False,
                )
            )

        if not fig.data:
            return ui.span()

        fig.update_yaxes(title_text="% responsive in top N", range=[0, 100])
        fig.update_layout(margin=dict(l=50, r=20, t=40, b=80))
        return ui.div(
            {"style": "margin-top: 1.5rem;"},
            ui.HTML(to_html(fig, include_plotlyjs=False, full_html=False)),
        )

    # ---------------------------------------------------------------------------
    # Tab 2: Compare Promoter Definitions
    # ---------------------------------------------------------------------------

    @render.ui
    def cp_promoter_table() -> ui.Tag:
        """
        Per-perturbation table comparing top-N % responsive across promoter sets.

        :trigger: ``_cp_data`` — re-renders when data changes.

        """
        if _results_stale():
            return ui.div(
                {"class": "empty-state"},
                ui.p("Click Execute Analysis to compute."),
            )
        df = _cp_data()
        p_dbs = active_perturbation_datasets()
        if not p_dbs:
            return ui.div(
                {"class": "empty-state"},
                ui.p("No perturbation datasets selected."),
            )
        if df.empty:
            return ui.div(
                {"class": "empty-state"},
                ui.p("No data for the selected datasets."),
            )

        try:
            included_ps = list(input.cp_included_promoter_sets())
        except Exception:
            included_ps = list(_PROMOTER_SET_ALIAS.keys())

        preset_name = _read_preset_name(input)
        _th_style = "padding: 6px 10px; text-align: right;"

        cards: list[ui.Tag] = []
        for p_db in p_dbs:
            p_label = PERTURBATION_LABEL_MAP.get(p_db, p_db)
            sub_p = df[df["perturbation_db"] == p_db]
            if sub_p.empty:
                continue

            thresh_label = get_responsiveness_label(preset_name, p_db)
            binding_base_labels = [
                b for b in _BINDING_ORDER
                if b in sub_p["binding_base_label"].unique()
            ]

            header_cells = [
                ui.tags.th(
                    "Binding Dataset",
                    style="padding: 6px 10px; text-align: left;",
                )
            ]
            for ps in included_ps:
                header_cells.append(ui.tags.th(ps, style=_th_style))

            data_rows: list[ui.Tag] = []
            for base_label in binding_base_labels:
                sub_b = sub_p[sub_p["binding_base_label"] == base_label]
                if sub_b.empty:
                    continue
                row_cells = [
                    ui.tags.td(
                        base_label,
                        style="padding: 6px 10px; text-align: left; white-space: nowrap;",
                    )
                ]
                for ps in included_ps:
                    sub_ps = sub_b[sub_b["promoter_set"] == ps]
                    if sub_ps.empty:
                        row_cells.append(
                            ui.tags.td("-", style="padding: 6px 10px; text-align: right;")
                        )
                    else:
                        per_reg = sub_ps.groupby("regulator_locus_tag")[
                            "percent_responsive"
                        ].median()
                        val = float(per_reg.median()) if not per_reg.empty else None
                        if val is not None and pd.notna(val):
                            row_cells.append(
                                ui.tags.td(f"{val:.1f}%", style=_cell_style(val))
                            )
                        else:
                            row_cells.append(
                                ui.tags.td("-", style="padding: 6px 10px; text-align: right;")
                            )
                data_rows.append(ui.tags.tr(*row_cells))

            if not data_rows:
                continue

            cards.append(
                ui.div(
                    {
                        "style": (
                            "flex: 1 1 0; border: 1px solid #ddd;"
                            " border-radius: 4px; overflow: hidden;"
                        )
                    },
                    ui.div(
                        {
                            "style": (
                                "padding: 6px 10px; font-weight: 600;"
                                " font-size: 0.9rem; background-color: #f5f5f5;"
                                " border-bottom: 1px solid #ddd;"
                            )
                        },
                        ui.tooltip(
                            ui.span(p_label),
                            f"Responsive threshold: {thresh_label}",
                        ),
                    ),
                    ui.tags.table(
                        {
                            "style": (
                                "border-collapse: collapse;"
                                " font-size: 0.9rem; width: 100%;"
                            )
                        },
                        ui.tags.thead(
                            {"style": "background-color: #f5f5f5;"},
                            ui.tags.tr(*header_cells),
                        ),
                        ui.tags.tbody(*data_rows),
                    ),
                )
            )

        if not cards:
            return ui.div(
                {"class": "empty-state"},
                ui.p("No data for the selected combination."),
            )

        return ui.div(
            {
                "style": (
                    "display: flex; flex-wrap: wrap; gap: 1.5rem; margin-top: 0.5rem;"
                )
            },
            *cards,
        )

    # ---------------------------------------------------------------------------
    # Tab 3: Compare Analysis Methods
    # ---------------------------------------------------------------------------

    @render.ui
    def cm_method_table() -> ui.Tag:
        """
        Per-perturbation table comparing top-N % responsive across scoring methods.

        :trigger: ``_cm_data`` — re-renders when data changes.

        """
        if _results_stale():
            return ui.div(
                {"class": "empty-state"},
                ui.p("Click Execute Analysis to compute."),
            )
        df = _cm_data()
        p_dbs = active_perturbation_datasets()
        if not p_dbs:
            return ui.div(
                {"class": "empty-state"},
                ui.p("No perturbation datasets selected."),
            )
        if df.empty:
            return ui.div(
                {"class": "empty-state"},
                ui.p("No data for the selected combination."),
            )

        preset_name = _read_preset_name(input)
        _th_style = "padding: 6px 10px; text-align: right;"

        variants_present = [
            v for v in SCORING_VARIANT_ORDER
            if v in df["scoring_variant"].unique()
        ]
        if not variants_present:
            return ui.span()

        cards: list[ui.Tag] = []
        for p_db in p_dbs:
            p_label = PERTURBATION_LABEL_MAP.get(p_db, p_db)
            sub_p = df[df["perturbation_db"] == p_db]
            if sub_p.empty:
                continue

            thresh_label = get_responsiveness_label(preset_name, p_db)
            method_base_labels = [
                b for b in list(dict.fromkeys(METHOD_BASE_LABEL_MAP.values()))
                if b in sub_p["scoring_variant"].map(
                    lambda v: METHOD_BASE_LABEL_MAP.get(
                        sub_p.loc[sub_p["scoring_variant"] == v, "binding_db"].iloc[0]
                        if not sub_p.loc[sub_p["scoring_variant"] == v].empty
                        else "", v
                    )
                ).unique()
            ]

            header_cells = [
                ui.tags.th(
                    "Scoring Variant",
                    style="padding: 6px 10px; text-align: left;",
                )
            ]
            for v in variants_present:
                color = SCORING_VARIANT_COLORS.get(v, "#888888")
                header_cells.append(
                    ui.tags.th(
                        v,
                        style=f"{_th_style} color: {color}; font-weight: 600;",
                    )
                )

            data_rows_cm: list[ui.Tag] = []
            # One row per unique binding base label (dataset)
            base_labels_in_data = list(
                dict.fromkeys(
                    METHOD_BASE_LABEL_MAP.get(b_db, b_db)
                    for b_db in df["binding_db"].unique()
                    if sub_p[sub_p["binding_db"] == b_db].shape[0] > 0
                )
            )
            for base_label in base_labels_in_data:
                row_cells = [
                    ui.tags.td(
                        base_label,
                        style="padding: 6px 10px; text-align: left; white-space: nowrap;",
                    )
                ]
                for v in variants_present:
                    sub_v = sub_p[sub_p["scoring_variant"] == v]
                    if sub_v.empty:
                        row_cells.append(
                            ui.tags.td("-", style="padding: 6px 10px; text-align: right;")
                        )
                    else:
                        per_reg = sub_v.groupby("regulator_locus_tag")[
                            "percent_responsive"
                        ].median()
                        val = float(per_reg.median()) if not per_reg.empty else None
                        if val is not None and pd.notna(val):
                            row_cells.append(
                                ui.tags.td(f"{val:.1f}%", style=_cell_style(val))
                            )
                        else:
                            row_cells.append(
                                ui.tags.td("-", style="padding: 6px 10px; text-align: right;")
                            )
                data_rows_cm.append(ui.tags.tr(*row_cells))

            if not data_rows_cm:
                continue

            cards.append(
                ui.div(
                    {
                        "style": (
                            "flex: 1 1 0; border: 1px solid #ddd;"
                            " border-radius: 4px; overflow: hidden;"
                        )
                    },
                    ui.div(
                        {
                            "style": (
                                "padding: 6px 10px; font-weight: 600;"
                                " font-size: 0.9rem; background-color: #f5f5f5;"
                                " border-bottom: 1px solid #ddd;"
                            )
                        },
                        ui.tooltip(
                            ui.span(p_label),
                            f"Responsive threshold: {thresh_label}",
                        ),
                    ),
                    ui.tags.table(
                        {
                            "style": (
                                "border-collapse: collapse;"
                                " font-size: 0.9rem; width: 100%;"
                            )
                        },
                        ui.tags.thead(
                            {"style": "background-color: #f5f5f5;"},
                            ui.tags.tr(*header_cells),
                        ),
                        ui.tags.tbody(*data_rows_cm),
                    ),
                )
            )

        if not cards:
            return ui.div(
                {"class": "empty-state"},
                ui.p("No data for the selected combination."),
            )

        return ui.div(
            {
                "style": (
                    "display: flex; flex-wrap: wrap; gap: 1.5rem; margin-top: 0.5rem;"
                )
            },
            *cards,
        )


__all__ = ["comparison_workspace_server"]
