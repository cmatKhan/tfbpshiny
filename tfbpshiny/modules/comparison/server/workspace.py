"""Workspace server for the Comparison module."""

from __future__ import annotations

import asyncio
import functools
from collections.abc import Callable
from logging import Logger
from typing import Any

import pandas as pd
import plotly.graph_objects as go
from labretriever import VirtualDB
from plotly.io import to_html
from shiny import reactive, render, req, ui
from shiny.reactive import extended_task
from shiny.ui import bind_task_button, input_task_button  # noqa: F401

from tfbpshiny.components import matrix_cell_button, sidebar_label
from tfbpshiny.modules.comparison.queries import (
    BINDING_BASE_LABEL_MAP,
    BINDING_CONFIGS,
    BINDING_LABEL_MAP,
    METHOD_BASE_LABEL_MAP,
    PEAKS_VARIANT_MAP,
    PERTURBATION_CONFIGS,
    PERTURBATION_LABEL_MAP,
    PROMOTER_SET_MAP,
    PROMOTER_VARIANT_PAIRS,
    SCORING_VARIANT_COLORS,
    SCORING_VARIANT_MAP,
    SCORING_VARIANT_ORDER,
    topn_all_pairs_sql,
)
from tfbpshiny.utils.perf import perf, reset_render_counts
from tfbpshiny.utils.topn_matrix import build_topn_matrix_ui
from tfbpshiny.utils.vdb_init import (
    DEFAULT_RESPONSIVENESS_PRESET,
    DEFAULT_RESPONSIVENESS_PRESETS,
    get_regulator_display_name,
    get_responsiveness_label,
)

# ---------------------------------------------------------------------------
# Display-order constants
# ---------------------------------------------------------------------------

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

# Binding datasets that can appear in the Compare Methods tab.
_METHODS_ELIGIBLE: frozenset[str] = frozenset(PEAKS_VARIANT_MAP)

# User-facing aliases for all four promoter set definitions.
_PROMOTER_SET_ALIAS: dict[str, str] = {
    "Kang": "Promoter Set 1 (Kang)",
    "Mindel": "Promoter Set 2 (Mindel)",
    "500bp": "Promoter Set 3 (500bp)",
    "Intergenic": "Promoter Set 4 (Intergenic)",
}

# Definitions sourced from brentlab_yeast_collection.yaml genome_resources.region_sets.
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


def _checkbox_group_with_disabled(
    input_id: str,
    choices: dict[str, str],
    selected: list[str],
    disabled: set[str],
) -> ui.Tag:
    """
    Build a checkbox group identical to ``ui.input_checkbox_group`` but with per-choice
    disabled support.

    Replicates Shiny's internal HTML structure (``name=input_id``,
    ``value=choice_value``, ``class="shiny-options-group"``), adding the
    HTML ``disabled`` attribute to any choice whose value is in ``disabled``.
    Disabled choices are also visually dimmed via inline opacity.

    :param input_id: The Shiny input ID (already namespaced by the caller).
    :param choices: Ordered ``{value: label}`` mapping.
    :param selected: Values that should be checked.
    :param disabled: Values that should be disabled (unchecked and non-interactive).
    :returns: A ``div.shiny-input-container`` tag matching Shiny's checkbox group.

    """
    option_tags: list[ui.Tag] = []
    for value, label in choices.items():
        is_disabled = value in disabled
        inp = ui.tags.input(
            type="checkbox",
            name=input_id,
            value=value,
            checked="checked" if (value in selected and not is_disabled) else None,
            disabled="disabled" if is_disabled else None,
        )
        option_tags.append(
            ui.div(
                ui.tags.label(
                    inp,
                    " ",
                    ui.span(label),
                    style="opacity: 0.45;" if is_disabled else None,
                ),
                class_="checkbox",
            )
        )
    return ui.div(
        ui.div(*option_tags, class_="shiny-options-group"),
        id=input_id,
        class_="shiny-input-checkboxgroup shiny-input-container",
    )


# ---------------------------------------------------------------------------
# Pure slice-labeling helpers (extracted from the old monolithic _run_analysis)
# ---------------------------------------------------------------------------


def _label_cd_slice(
    raw: pd.DataFrame,
    cd_resolved: list[tuple[str, str]],
    p_db: str,
    reg_labels: dict[str, str],
) -> pd.DataFrame:
    """
    Label a raw Compare Datasets query result for one perturbation dataset.

    Filters ``raw`` to rows matching each ``(resolved_db, p_db)`` pair, adds
    display columns, and concatenates.

    :param raw: Raw ``topn_all_pairs_sql`` result (contains ``pair_key`` column).
    :param cd_resolved: Ordered ``(resolved_db, primary_b_db)`` pairs for all
        binding datasets in the run.
    :param p_db: Perturbation dataset db_name for this slice.
    :param reg_labels: Mapping from ``regulator_locus_tag`` to display name.
    :returns: Labeled DataFrame for this perturbation slice, or empty DataFrame.

    """
    rows: list[pd.DataFrame] = []
    for resolved, b_db in cd_resolved:
        pair_key = f"{resolved}__{p_db}"
        sub = (
            raw[raw["pair_key"] == pair_key]
            .drop(columns=["pair_key"])
            .reset_index(drop=True)
            .copy()
        )
        if sub.empty:
            continue
        sub["binding_db"] = b_db
        sub["binding_label"] = BINDING_LABEL_MAP.get(b_db, b_db)
        sub["perturbation_db"] = p_db
        sub["perturbation_source"] = PERTURBATION_LABEL_MAP.get(p_db, p_db)
        sub["regulator_label"] = (
            sub["regulator_locus_tag"]
            .map(reg_labels)
            .fillna(sub["regulator_locus_tag"])
        )
        sub["percent_responsive"] = sub["responsive_ratio"] * 100
        rows.append(sub)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _label_cp_slice(
    raw: pd.DataFrame,
    expanded: list[str],
    p_db: str,
    reg_labels: dict[str, str],
) -> pd.DataFrame:
    """
    Label a raw Compare Promoters query result for one perturbation dataset.

    :param raw: Raw ``topn_all_pairs_sql`` result.
    :param expanded: Ordered list of binding db_names (Kang and/or Mindel variants).
    :param p_db: Perturbation dataset db_name for this slice.
    :param reg_labels: Mapping from ``regulator_locus_tag`` to display name.
    :returns: Labeled DataFrame for this perturbation slice, or empty DataFrame.

    """
    rows: list[pd.DataFrame] = []
    for b_db in expanded:
        pair_key = f"{b_db}__{p_db}"
        sub = (
            raw[raw["pair_key"] == pair_key]
            .drop(columns=["pair_key"])
            .reset_index(drop=True)
            .copy()
        )
        if sub.empty:
            continue
        sub["binding_base_label"] = BINDING_BASE_LABEL_MAP.get(b_db, b_db)
        sub["promoter_set"] = PROMOTER_SET_MAP.get(b_db, "Kang")
        sub["perturbation_source"] = PERTURBATION_LABEL_MAP.get(p_db, p_db)
        sub["regulator_label"] = (
            sub["regulator_locus_tag"]
            .map(reg_labels)
            .fillna(sub["regulator_locus_tag"])
        )
        sub["percent_responsive"] = sub["responsive_ratio"] * 100
        rows.append(sub)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _label_cm_slice(
    raw: pd.DataFrame,
    all_method_dbs: list[str],
    p_db: str,
    reg_labels: dict[str, str],
) -> pd.DataFrame:
    """
    Label a raw Compare Methods query result for one perturbation dataset.

    :param raw: Raw ``topn_all_pairs_sql`` result.
    :param all_method_dbs: Ordered list of method variant db_names.
    :param p_db: Perturbation dataset db_name for this slice.
    :param reg_labels: Mapping from ``regulator_locus_tag`` to display name.
    :returns: Labeled DataFrame for this perturbation slice, or empty DataFrame.

    """
    rows: list[pd.DataFrame] = []
    for b_db in all_method_dbs:
        pair_key = f"{b_db}__{p_db}"
        sub = (
            raw[raw["pair_key"] == pair_key]
            .drop(columns=["pair_key"])
            .reset_index(drop=True)
            .copy()
        )
        if sub.empty:
            continue
        sub["scoring_variant"] = SCORING_VARIANT_MAP.get(b_db, b_db)
        sub["perturbation_source"] = PERTURBATION_LABEL_MAP.get(p_db, p_db)
        sub["regulator_label"] = (
            sub["regulator_locus_tag"]
            .map(reg_labels)
            .fillna(sub["regulator_locus_tag"])
        )
        sub["percent_responsive"] = sub["responsive_ratio"] * 100
        rows.append(sub)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def comparison_workspace_server(
    input: Any,
    output: Any,
    session: Any,
    active_binding_datasets: reactive.Calc_[list[str]],
    active_perturbation_datasets: reactive.Calc_[list[str]],
    dataset_filters: reactive.Value[dict[str, Any]],
    vdb: VirtualDB,
    logger: Logger,
    active_tab: reactive.Calc_[str] | None = None,
    materialize_ready: Callable[[], bool] | None = None,
) -> None:
    """Render the Comparison workspace: Compare Datasets, Promoters, Methods."""

    session.on_flush(lambda: reset_render_counts(session.id))

    def _timed_render(label: str) -> Any:
        """
        Decorator that wraps a ``render`` function in a :func:`perf` timing block.

        Applied beneath ``@render.ui`` so the rendered output id (derived from the
        function name) is preserved via ``functools.wraps``.

        :param label: perf label for the render, e.g. ``"cd_matrix_container"``.

        """

        def deco(fn: Any) -> Any:
            @functools.wraps(fn)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                with perf(session.id, "comparison.workspace", label):
                    return fn(*args, **kwargs)

            return wrapper

        return deco

    _available_datasets: frozenset[str] = frozenset(vdb.get_datasets())
    # All non-primary promoter-set variants (Mindel, 500bp, Intergenic) — excluded
    # from the primary binding lists used by all three subtabs.
    _variant_dbs: frozenset[str] = (
        frozenset(db for variants in PROMOTER_VARIANT_PAIRS.values() for db in variants)
        & _available_datasets
    )

    _reg_df = get_regulator_display_name(vdb)
    _reg_labels: dict[str, str] = dict(
        zip(_reg_df["regulator_locus_tag"], _reg_df["display_name"])
    )

    display_names: dict[str, str] = {
        db: vdb.get_tags(db).get("display_name", db) for db in vdb.get_datasets()
    }

    # All primary binding datasets known to this VDB (excludes Mindel/peaks).
    _all_primary_binding: list[str] = sorted(
        db
        for db in vdb.get_datasets()
        if db in BINDING_CONFIGS
        and db not in _variant_dbs
        and db not in frozenset().union(*PEAKS_VARIANT_MAP.values())
    )
    _all_perturbation: list[str] = sorted(
        db for db in vdb.get_datasets() if db in PERTURBATION_CONFIGS
    )

    # Selected row/column in the Compare Datasets matrix.
    cd_selected_binding: reactive.Value[str | None] = reactive.value(None)
    cd_selected_perturbation: reactive.Value[str | None] = reactive.value(None)

    # Snapshot of the sidebar inputs at the last successful Execute, keyed by
    # inner tab.
    _last_run_snapshot: reactive.Value[dict[str, tuple]] = reactive.value({})

    # True whenever results are out of date and Execute Analysis must be re-run.
    _results_stale: reactive.Value[bool] = reactive.value(True)

    # Config-keyed result cache: maps a full sidebar-config key to the assembled
    # per-tab result dict. Re-running an identical config returns instantly.
    _config_cache: dict[tuple, dict] = {}
    _CONFIG_CACHE_MAX = 16

    # ---------------------------------------------------------------------------
    # Per-perturbation reactive slots (progressive loading independence)
    # ---------------------------------------------------------------------------
    # Each dict maps perturbation db_name -> reactive.Value holding the labeled
    # slice DataFrame for that perturbation, or None when not yet computed.
    # Separate dicts per tab so a cp slice never re-triggers a cm output.
    _cd_slot: dict[str, reactive.Value[pd.DataFrame | None]] = {
        p: reactive.value(None) for p in _all_perturbation
    }
    _cp_slot: dict[str, reactive.Value[pd.DataFrame | None]] = {
        p: reactive.value(None) for p in _all_perturbation
    }
    _cm_slot: dict[str, reactive.Value[pd.DataFrame | None]] = {
        p: reactive.value(None) for p in _all_perturbation
    }

    # Run-level state --------------------------------------------------------
    # Bumped on every Execute; slice results carrying a stale epoch are dropped.
    _run_epoch: reactive.Value[int] = reactive.value(0)

    # Per-tab run metadata — one entry per tab, persists across runs on other
    # tabs so each scaffold can show its prior results when switching back.
    # Keys per entry: perturbation_list, cd_resolved, expanded, all_method_dbs,
    #   cd_binding_datasets, cp_included_promoter_sets, cm_binding_db, config_key,
    #   top_n, preset, filters.
    _run_meta_by_tab: reactive.Value[dict[str, dict]] = reactive.value({})

    # Remaining perturbations to compute in the current run.
    _slice_queue: reactive.Value[list[str]] = reactive.value([])

    # ---------------------------------------------------------------------------
    # Helper: derive active tab
    # ---------------------------------------------------------------------------

    def _inner_tab() -> str:
        try:
            return str(input.comparison_inner_tabs())
        except Exception:
            return "Compare Datasets"

    # ---------------------------------------------------------------------------
    # Snapshot
    # ---------------------------------------------------------------------------

    def _snapshot_current() -> tuple:
        """
        Return a hashable representation of the *active* subtab's inputs.

        Scoped to the current inner tab (plus the shared ``top_n`` and dataset
        filters) rather than every subtab's inputs. Shiny retains an input's
        value after its control is unmounted, so including the other subtabs'
        inputs made the snapshot change merely by visiting those subtabs — which
        falsely re-activated the Execute button on returning to an already-run
        subtab. Comparing only the active subtab's inputs keeps the pending
        state stable across subtab navigation.

        """
        try:
            filters_repr = repr(
                sorted((k, repr(v)) for k, v in dataset_filters().items())
            )
        except Exception:
            filters_repr = ""

        def _safe(fn: Any) -> Any:
            try:
                v = fn()
                if hasattr(v, "__iter__") and not isinstance(v, str):
                    return tuple(sorted(v))
                return v
            except Exception:
                return None

        tab = _inner_tab()
        base: tuple = (
            tab,
            input.top_n(),
            filters_repr,
            _safe(input.responsiveness_preset),
        )
        if tab == "Compare Datasets":
            return base + (
                _safe(input.cd_binding_method),
                _safe(input.cd_promoter_set),
            )
        if tab == "Compare Promoter Definitions":
            return base + (_safe(input.cp_included_promoter_sets),)
        if tab == "Compare Analysis Methods":
            return base + (
                _safe(input.cm_binding_dataset),
                _safe(input.cm_promoter_set),
            )
        return base

    # ---------------------------------------------------------------------------
    # Staleness: mark results stale when committed dataset filters change
    # ---------------------------------------------------------------------------

    @reactive.effect
    def _mark_stale_on_filter_change() -> None:
        """
        Mark results stale whenever committed dataset filters change.

        Also resets all per-perturbation slots and bumps the run epoch so any in-flight
        slice results are discarded.

        :trigger dataset_filters: fires whenever Apply Changes is pressed in     Select
        Datasets.

        """
        dataset_filters()
        with reactive.isolate():
            if not _results_stale():
                logger.debug("_mark_stale_on_filter_change: marking results stale")
                _results_stale.set(True)
                _last_run_snapshot.set({})
                _run_epoch.set(_run_epoch() + 1)
                _run_meta_by_tab.set({})
                _slice_queue.set([])
                for p in _all_perturbation:
                    _cd_slot[p].set(None)
                    _cp_slot[p].set(None)
                    _cm_slot[p].set(None)

    # ---------------------------------------------------------------------------
    # Pending button style
    # ---------------------------------------------------------------------------

    @output(suspend_when_hidden=False)
    @render.ui
    def execute_pending_style() -> ui.Tag:
        """
        Dims the Execute button when no changes are pending.

        :trigger _snapshot inputs: re-fires on any sidebar change. :trigger
        _last_run_snapshot: re-fires after Execute. :trigger _results_stale:
        re-fires when dataset filters change.

        """
        if _results_stale():
            return ui.span()
        current = _snapshot_current()
        last = _last_run_snapshot().get(_inner_tab())
        has_pending = (last is None) or (current != last)
        if has_pending:
            return ui.span()
        btn_id = session.ns("execute_analysis")
        return ui.tags.style(
            f"#{btn_id} {{ opacity: 0.35; pointer-events: none; cursor: not-allowed; }}"
        )

    # ---------------------------------------------------------------------------
    # Tab-specific sidebar controls
    # ---------------------------------------------------------------------------

    @render.ui
    def tab_specific_controls() -> ui.Tag:
        """
        Render sidebar controls appropriate for the active inner tab.

        :trigger input.comparison_inner_tabs: re-renders when the tab changes. :trigger
        active_binding_datasets: re-renders when dataset selection changes. :trigger
        active_perturbation_datasets: re-renders when dataset selection changes.

        """
        tab = _inner_tab()
        binding_primary = [
            db
            for db in active_binding_datasets()
            if db in BINDING_CONFIGS
            and db not in _variant_dbs
            and db not in frozenset().union(*PEAKS_VARIANT_MAP.values())
        ]
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
            eligible = [db for db in binding_primary if db in _METHODS_ELIGIBLE]
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

    # ---------------------------------------------------------------------------
    # Per-perturbation slice task
    # ---------------------------------------------------------------------------

    @bind_task_button(button_id="execute_analysis")
    @extended_task
    async def _run_slice(
        epoch: int,
        tab: str,
        p_db: str,
        pairs: list[tuple[str, str]],
        filters: dict,
        top_n: int,
        preset: dict,
        # Extra labeling context passed through so _slice_driver can label without
        # re-reading reactive state.
        cd_resolved: list[tuple[str, str]],
        expanded: list[str],
        all_method_dbs: list[str],
    ) -> dict:
        """
        Compute top-N responsive ratios for one perturbation dataset, off-thread.

        Returns a result dict containing the epoch, tab, perturbation db_name, and
        labeled slice DataFrame. The epoch is checked by ``_slice_driver`` to discard
        results from superseded runs.

        :returns: Dict with keys ``epoch``, ``tab``, ``p_db``, ``df``.

        """
        perf_label = f"{tab[:2].lower()}p_topn_all_pairs_sql"
        try:
            with perf(session.id, "comparison.workspace", perf_label, kind="data"):
                raw = await asyncio.to_thread(
                    topn_all_pairs_sql, vdb, pairs, filters, top_n, preset
                )
        except Exception as exc:
            logger.error("_run_slice %s/%s failed: %s", tab, p_db, exc, exc_info=True)
            raw = pd.DataFrame()

        if raw.empty or "pair_key" not in raw.columns:
            return {"epoch": epoch, "tab": tab, "p_db": p_db, "df": pd.DataFrame()}

        if tab == "Compare Datasets":
            df = _label_cd_slice(raw, cd_resolved, p_db, _reg_labels)
        elif tab == "Compare Promoter Definitions":
            df = _label_cp_slice(raw, expanded, p_db, _reg_labels)
        else:
            df = _label_cm_slice(raw, all_method_dbs, p_db, _reg_labels)

        return {"epoch": epoch, "tab": tab, "p_db": p_db, "df": df}

    # ---------------------------------------------------------------------------
    # Execute handler
    # ---------------------------------------------------------------------------

    @reactive.effect
    @reactive.event(input.execute_analysis)
    def _on_execute() -> None:
        """
        Collect sidebar state, check the config cache, then start the first slice.

        :trigger input.execute_analysis: fires when Execute Analysis is clicked.

        """
        # Gate vdb access until background materialization finishes; the DuckDB
        # connection is not safe to touch while materialization mutates it.
        if materialize_ready is not None:
            req(materialize_ready())
        tab = _inner_tab()
        top_n = input.top_n()
        try:
            preset_name = input.responsiveness_preset()
        except Exception:
            preset_name = DEFAULT_RESPONSIVENESS_PRESET
        preset = DEFAULT_RESPONSIVENESS_PRESETS.get(preset_name, {"*": (0.0, 0.05)})
        filters = dataset_filters()

        def _safe_str(fn: Any, fallback: str = "") -> str:
            try:
                return str(fn()) if fn() else fallback
            except Exception:
                return fallback

        binding_primary = [
            db
            for db in active_binding_datasets()
            if db in BINDING_CONFIGS
            and db not in _variant_dbs
            and db not in frozenset().union(*PEAKS_VARIANT_MAP.values())
        ]
        pert_all = [
            db for db in active_perturbation_datasets() if db in PERTURBATION_CONFIGS
        ]

        # Resolve per-tab inputs.
        cd_method = _safe_str(input.cd_binding_method, "Promoter Enrichment")
        cd_promoter_set = _safe_str(input.cd_promoter_set, "Kang")

        try:
            cp_included_promoter_sets = list(input.cp_included_promoter_sets() or [])
        except Exception:
            cp_included_promoter_sets = list(_PROMOTER_SET_ALIAS.keys())

        eligible = [db for db in binding_primary if db in _METHODS_ELIGIBLE]
        cm_binding_db = _safe_str(input.cm_binding_dataset)
        if not cm_binding_db and eligible:
            cm_binding_db = eligible[0]
        try:
            cm_promoter_sets = list(input.cm_promoter_set() or [])
        except Exception:
            cm_promoter_sets = ["Kang"]

        # All active datasets are always included — no per-tab checkboxes.
        cd_included_binding = binding_primary
        cd_included_perturbation = pert_all
        cp_included_binding = binding_primary
        cp_included_perturbation = pert_all
        cm_included_perturbation = pert_all

        # Build the config cache key.
        config_key = (
            tab,
            top_n,
            preset_name,
            repr(sorted((k, repr(v)) for k, v in filters.items())),
            repr(sorted((k, repr(v)) for k, v in preset.items())),
            cd_method,
            cd_promoter_set,
            tuple(cd_included_binding),
            tuple(cd_included_perturbation),
            tuple(cp_included_binding),
            tuple(cp_included_perturbation),
            tuple(cp_included_promoter_sets),
            cm_binding_db,
            tuple(cm_promoter_sets),
            tuple(cm_included_perturbation),
        )

        cached = _config_cache.get(config_key)
        if cached is not None:
            # Cache hit: fill all slots from the cached assembled DataFrames at once
            # and mark not stale — no slicing needed.
            logger.debug("_on_execute: cache hit for tab=%s", tab)
            if tab == "Compare Datasets":
                cd_df: pd.DataFrame = cached.get("cd_data", pd.DataFrame())
                for p_db in cd_included_perturbation:
                    if p_db in _cd_slot:
                        sub = (
                            cd_df[cd_df["perturbation_db"] == p_db]
                            if not cd_df.empty
                            else pd.DataFrame()
                        )
                        _cd_slot[p_db].set(sub if not sub.empty else pd.DataFrame())
            elif tab == "Compare Promoter Definitions":
                cp_df: pd.DataFrame = cached.get("cp_topn_data", pd.DataFrame())
                for p_db in cp_included_perturbation:
                    if p_db in _cp_slot:
                        p_label = PERTURBATION_LABEL_MAP.get(p_db, p_db)
                        sub = (
                            cp_df[cp_df["perturbation_source"] == p_label]
                            if not cp_df.empty
                            else pd.DataFrame()
                        )
                        _cp_slot[p_db].set(sub if not sub.empty else pd.DataFrame())
            else:
                cm_df: pd.DataFrame = cached.get("cm_topn_data", pd.DataFrame())
                for p_db in cm_included_perturbation:
                    if p_db in _cm_slot:
                        p_label = PERTURBATION_LABEL_MAP.get(p_db, p_db)
                        sub = (
                            cm_df[cm_df["perturbation_source"] == p_label]
                            if not cm_df.empty
                            else pd.DataFrame()
                        )
                        _cm_slot[p_db].set(sub if not sub.empty else pd.DataFrame())
            _results_stale.set(False)
            _last_run_snapshot.set({**_last_run_snapshot(), tab: _snapshot_current()})
            return

        # --- Resolve per-tab pair structures for the slice driver ---

        # Compare Datasets: resolve binding db_names to correct promoter-set variant.
        # For Peaks, there is only one peaks variant per primary dataset.
        # For Promoter Enrichment, map the selected promoter set key to the variant
        # db_name: "Kang" -> primary db, others -> index into PROMOTER_VARIANT_PAIRS.
        _ps_to_variant_index: dict[str, int] = {
            "Mindel": 0,
            "500bp": 1,
            "Intergenic": 2,
        }

        def _resolve_cd_db(b_db: str) -> str | None:
            if cd_method == "Peaks":
                variants = PEAKS_VARIANT_MAP.get(b_db, [])
                return variants[0] if variants else None
            else:
                if cd_promoter_set == "Kang":
                    return b_db
                idx = _ps_to_variant_index.get(cd_promoter_set)
                if idx is None:
                    return b_db
                all_variants = PROMOTER_VARIANT_PAIRS.get(b_db, [])
                if idx < len(all_variants) and all_variants[idx] in _available_datasets:
                    return all_variants[idx]
                return b_db

        cd_resolved: list[tuple[str, str]] = [
            (resolved, b_db)
            for b_db in cd_included_binding
            for resolved in [_resolve_cd_db(b_db)]
            if resolved and resolved in BINDING_CONFIGS
        ]

        # Compare Promoters: expand each primary binding db to all selected
        # promoter-set variants. "Kang" maps to the primary db itself; others
        # map to the corresponding entry in PROMOTER_VARIANT_PAIRS.
        expanded: list[str] = []
        for b_db in cp_included_binding:
            if "Kang" in cp_included_promoter_sets:
                expanded.append(b_db)
            for ps_key, idx in _ps_to_variant_index.items():
                if ps_key in cp_included_promoter_sets:
                    all_variants = PROMOTER_VARIANT_PAIRS.get(b_db, [])
                    if (
                        idx < len(all_variants)
                        and all_variants[idx] in _available_datasets
                    ):
                        expanded.append(all_variants[idx])

        # Compare Methods: include the primary (Kang) db and/or the selected
        # promoter-set variants, plus the peaks variant.
        peaks = [
            pk
            for pk in PEAKS_VARIANT_MAP.get(cm_binding_db, [])
            if pk in _available_datasets
        ]
        all_variants_cm = PROMOTER_VARIANT_PAIRS.get(cm_binding_db, [])
        all_method_dbs = (
            ([cm_binding_db] if cm_binding_db and "Kang" in cm_promoter_sets else [])
            + [
                all_variants_cm[idx]
                for ps_key, idx in _ps_to_variant_index.items()
                if ps_key in cm_promoter_sets
                and idx < len(all_variants_cm)
                and all_variants_cm[idx] in _available_datasets
            ]
            + peaks
        )
        all_method_dbs = [
            b
            for b in all_method_dbs
            if b in METHOD_BASE_LABEL_MAP and b in BINDING_CONFIGS
        ]

        # Determine the ordered perturbation list for this tab.
        if tab == "Compare Datasets":
            pert_list = [
                p for p in cd_included_perturbation if p in PERTURBATION_CONFIGS
            ]
        elif tab == "Compare Promoter Definitions":
            pert_list = [
                p for p in cp_included_perturbation if p in PERTURBATION_CONFIGS
            ]
        else:
            pert_list = [
                p for p in cm_included_perturbation if p in PERTURBATION_CONFIGS
            ]

        if not pert_list:
            return

        # Bump epoch and reset this tab's slots.
        new_epoch = _run_epoch() + 1
        _run_epoch.set(new_epoch)
        for p in _all_perturbation:
            if tab == "Compare Datasets":
                _cd_slot[p].set(None)
            elif tab == "Compare Promoter Definitions":
                _cp_slot[p].set(None)
            else:
                _cm_slot[p].set(None)

        _results_stale.set(False)
        _last_run_snapshot.set({**_last_run_snapshot(), tab: _snapshot_current()})

        # Store per-tab run metadata so each scaffold retains its own run config
        # when the user switches to a different subtab and back.
        _run_meta_by_tab.set(
            {
                **_run_meta_by_tab(),
                tab: {
                    "perturbation_list": pert_list,
                    "cd_resolved": cd_resolved,
                    "expanded": expanded,
                    "all_method_dbs": all_method_dbs,
                    "cd_binding_datasets": [b_db for _, b_db in cd_resolved],
                    "cp_included_promoter_sets": cp_included_promoter_sets,
                    "cm_binding_db": cm_binding_db,
                    "config_key": config_key,
                    "top_n": top_n,
                    "preset": preset,
                    "filters": filters,
                },
            }
        )
        _slice_queue.set(pert_list[1:])

        # Launch the first slice.
        p0 = pert_list[0]
        _launch_slice(
            new_epoch,
            tab,
            p0,
            cd_resolved,
            expanded,
            all_method_dbs,
            top_n,
            preset,
            filters,
        )

    def _launch_slice(
        epoch: int,
        tab: str,
        p_db: str,
        cd_resolved: list[tuple[str, str]],
        expanded: list[str],
        all_method_dbs: list[str],
        top_n: int,
        preset: dict,
        filters: dict,
    ) -> None:
        """Build the pairs list for one perturbation and invoke _run_slice."""
        if tab == "Compare Datasets":
            pairs = [
                (resolved, p_db)
                for resolved, _ in cd_resolved
                if p_db in PERTURBATION_CONFIGS
            ]
        elif tab == "Compare Promoter Definitions":
            pairs = [
                (b_db, p_db)
                for b_db in expanded
                if b_db in BINDING_CONFIGS and p_db in PERTURBATION_CONFIGS
            ]
        else:
            pairs = [
                (b_db, p_db) for b_db in all_method_dbs if p_db in PERTURBATION_CONFIGS
            ]

        if not pairs:
            # Nothing to compute for this perturbation — fake a success by setting
            # the slot directly and advancing the queue.
            _advance_queue(epoch, tab, p_db, pd.DataFrame())
            return

        _run_slice.invoke(
            epoch,
            tab,
            p_db,
            pairs,
            filters,
            top_n,
            preset,
            cd_resolved,
            expanded,
            all_method_dbs,
        )

    def _advance_queue(epoch: int, tab: str, p_db: str, df: pd.DataFrame) -> None:
        """
        Store a completed slice result and launch the next slice if the queue is non-
        empty.

        Also assembles and caches the full result when the queue empties.

        """
        with reactive.isolate():
            current_epoch = _run_epoch()
        if epoch != current_epoch:
            return

        # Set the appropriate slot.
        if tab == "Compare Datasets":
            if p_db in _cd_slot:
                _cd_slot[p_db].set(df)
        elif tab == "Compare Promoter Definitions":
            if p_db in _cp_slot:
                _cp_slot[p_db].set(df)
        else:
            if p_db in _cm_slot:
                _cm_slot[p_db].set(df)

        with reactive.isolate():
            queue = list(_slice_queue())
            meta = _run_meta_by_tab().get(tab, {})

        if not queue:
            # All slices done — assemble and cache the full result.
            _assemble_and_cache(epoch, tab, meta)
            return

        next_p = queue[0]
        _slice_queue.set(queue[1:])
        _launch_slice(
            epoch,
            tab,
            next_p,
            meta.get("cd_resolved", []),
            meta.get("expanded", []),
            meta.get("all_method_dbs", []),
            meta.get("top_n", 25),
            meta.get("preset", {}),
            meta.get("filters", {}),
        )

    def _assemble_and_cache(epoch: int, tab: str, meta: dict) -> None:
        """Concatenate all slot DataFrames for this tab and store in _config_cache."""
        with reactive.isolate():
            if epoch != _run_epoch():
                return
            config_key = meta.get("config_key")
            if config_key is None:
                return

            pert_list: list[str] = meta.get("perturbation_list", [])

            if tab == "Compare Datasets":
                parts = [_cd_slot[p]() for p in pert_list if p in _cd_slot]
                full_df = (
                    pd.concat(
                        [f for f in parts if f is not None and not f.empty],
                        ignore_index=True,
                    )
                    if parts
                    else pd.DataFrame()
                )
                result = {
                    "tab": tab,
                    "cd_data": full_df,
                    "cd_binding_datasets": meta.get("cd_binding_datasets", []),
                    "cd_perturbation_datasets": pert_list,
                    "cp_topn_data": pd.DataFrame(),
                    "cp_included_promoter_sets": [],
                    "cm_topn_data": pd.DataFrame(),
                    "cm_binding_db": None,
                }
            elif tab == "Compare Promoter Definitions":
                parts = [_cp_slot[p]() for p in pert_list if p in _cp_slot]
                full_df = (
                    pd.concat(
                        [f for f in parts if f is not None and not f.empty],
                        ignore_index=True,
                    )
                    if parts
                    else pd.DataFrame()
                )
                result = {
                    "tab": tab,
                    "cd_data": pd.DataFrame(),
                    "cd_binding_datasets": [],
                    "cd_perturbation_datasets": [],
                    "cp_topn_data": full_df,
                    "cp_included_promoter_sets": meta.get(
                        "cp_included_promoter_sets", []
                    ),
                    "cm_topn_data": pd.DataFrame(),
                    "cm_binding_db": None,
                }
            else:
                parts = [_cm_slot[p]() for p in pert_list if p in _cm_slot]
                full_df = (
                    pd.concat(
                        [f for f in parts if f is not None and not f.empty],
                        ignore_index=True,
                    )
                    if parts
                    else pd.DataFrame()
                )
                result = {
                    "tab": tab,
                    "cd_data": pd.DataFrame(),
                    "cd_binding_datasets": [],
                    "cd_perturbation_datasets": [],
                    "cp_topn_data": pd.DataFrame(),
                    "cp_included_promoter_sets": [],
                    "cm_topn_data": full_df,
                    "cm_binding_db": meta.get("cm_binding_db"),
                }

            _config_cache[config_key] = result
            if len(_config_cache) > _CONFIG_CACHE_MAX:
                _config_cache.pop(next(iter(_config_cache)))

    # ---------------------------------------------------------------------------
    # Slice driver: fires when _run_slice completes, advances the queue
    # ---------------------------------------------------------------------------

    @reactive.effect
    def _slice_driver() -> None:
        """
        Receive a completed slice result and chain to the next perturbation.

        :trigger _run_slice.status: fires on every task status change.

        """
        if _run_slice.status() != "success":
            return
        result = _run_slice.result()
        epoch: int = result["epoch"]
        tab: str = result["tab"]
        p_db: str = result["p_db"]
        df: pd.DataFrame = result["df"]
        _advance_queue(epoch, tab, p_db, df)

    # ---------------------------------------------------------------------------
    # Status render
    # ---------------------------------------------------------------------------

    @output(suspend_when_hidden=False)
    @render.ui
    def analysis_status() -> ui.Tag:
        """
        Show per-slice progress while computing.

        :trigger _run_slice.status: re-renders when the task state changes. :trigger
        _slice_queue: re-renders as slices complete.

        """
        status = _run_slice.status()
        if status == "running":
            with reactive.isolate():
                # Peek at the currently-running tab's meta via the slice queue's tab.
                # The slice task result carries the tab, but status=="running" means
                # no result yet — use the most recently set tab entry.
                all_meta = _run_meta_by_tab()
                # Find the tab whose meta was set most recently (last key in insertion
                # order that has a perturbation_list matching the queue length).
                queue = _slice_queue()
                meta: dict = next(
                    (
                        v
                        for v in reversed(list(all_meta.values()))
                        if v.get("perturbation_list")
                    ),
                    {},
                )
            total = len(meta.get("perturbation_list", []))
            remaining = len(queue) + 1  # +1 for the currently running slice
            done = total - remaining
            return ui.div(
                {"class": "empty-state"},
                ui.p(
                    f"Computing perturbation {done + 1} of {total}. "
                    "Thank you for your patience."
                ),
            )
        if status == "error":
            return ui.div(
                {"class": "empty-state"},
                ui.p(f"Error: {_run_slice.error()}"),
            )
        return ui.span()

    # ---------------------------------------------------------------------------
    # Helper: table cell style
    # ---------------------------------------------------------------------------

    def _cell_style(val: float) -> str:
        """HSL green scale: 0% -> white, 100% -> full green."""
        clamped = max(0.0, min(100.0, val))
        lightness = 100 - clamped * 0.5
        return (
            f"background-color: hsl(120, 60%, {lightness:.0f}%);"
            " padding: 6px 10px; text-align: right;"
        )

    # ===========================================================================
    # Tab 1: Compare Datasets
    # ===========================================================================

    # All possible (b_db, p_db) combinations at init time for pre-registering effects.
    _all_cd_pairs: list[tuple[str, str]] = [
        (b_db, p_db) for b_db in _all_primary_binding for p_db in _all_perturbation
    ]

    def _make_cd_cell_effect(b_db: str, p_db: str) -> None:
        btn_id = f"topncell_{b_db}__{p_db}"

        @reactive.effect
        @reactive.event(input[btn_id])
        def _on_cell() -> None:
            cd_selected_binding.set(b_db)
            cd_selected_perturbation.set(None)

    def _make_cd_row_effect(b_db: str) -> None:
        btn_id = f"topnrow_{b_db}"

        @reactive.effect
        @reactive.event(input[btn_id])
        def _on_row() -> None:
            cd_selected_binding.set(b_db)
            cd_selected_perturbation.set(None)

    def _make_cd_col_effect(p_db: str) -> None:
        btn_id = f"topncol_{p_db}"

        @reactive.effect
        @reactive.event(input[btn_id])
        def _on_col() -> None:
            cd_selected_perturbation.set(p_db)
            cd_selected_binding.set(None)

    for _b, _p in _all_cd_pairs:
        _make_cd_cell_effect(_b, _p)
    for _b in _all_primary_binding:
        _make_cd_row_effect(_b)
    for _p in _all_perturbation:
        _make_cd_col_effect(_p)

    # Pre-register one render per (b_db, p_db) cell in the universe.
    # Each reads only _cd_slot[p_db], so only column p_db's cells re-render when
    # that perturbation's slice lands.
    def _make_cd_cell_render(b_db: str, p_db: str) -> None:
        @output(id=f"cdcell_{b_db}__{p_db}")
        @render.ui
        def _cdcell() -> ui.Tag:
            df = _cd_slot[p_db]()
            sel_b = cd_selected_binding()
            sel_p = cd_selected_perturbation()
            cell_active = (sel_b == b_db) or (sel_p == p_db)
            if df is None:
                # Slice not yet computed — show placeholder.
                return ui.span(
                    {"class": "cd-cell-loading", "aria-label": "loading"},
                    "...",
                )
            if df.empty:
                return matrix_cell_button(session.ns(f"topncell_{b_db}__{p_db}"), "—")
            sub = df[df["binding_db"] == b_db]
            if sub.empty:
                return matrix_cell_button(session.ns(f"topncell_{b_db}__{p_db}"), "—")
            per_reg = sub.groupby("regulator_locus_tag")["percent_responsive"].median()
            val = float(per_reg.median()) if not per_reg.empty else None
            label = f"{val:.1f}%" if val is not None and not pd.isna(val) else "—"
            _ = cell_active  # read for dependency but active class is on the td
            return matrix_cell_button(session.ns(f"topncell_{b_db}__{p_db}"), label)

    for _b, _p in _all_cd_pairs:
        _make_cd_cell_render(_b, _p)

    @output(suspend_when_hidden=False)
    @render.ui
    @_timed_render("cd_matrix_container")
    def cd_matrix_container() -> ui.Tag:
        """
        Binding x perturbation matrix scaffold with per-cell output_ui slots.

        Renders the header/row structure once per Execute (depends on _run_meta,
        not slot data). Each interactive cell is a ``ui.output_ui`` slot filled
        independently by the pre-registered ``cdcell_{b}__{p}`` render.

        :trigger _run_meta: re-renders when a new Execute starts (new run config).
        :trigger _results_stale: re-renders when results are invalidated.
        :trigger cd_selected_binding: re-renders to move row highlight.
        :trigger cd_selected_perturbation: re-renders to move column highlight.

        """
        if _results_stale():
            return ui.div(
                {"class": "empty-state"},
                ui.p("Click Execute Analysis to compute."),
            )
        meta = _run_meta_by_tab().get("Compare Datasets", {})
        if not meta:
            return ui.div(
                {"class": "empty-state"},
                ui.p("Click Execute Analysis to compute."),
            )

        b_datasets: list[str] = meta.get("cd_binding_datasets", [])
        p_datasets: list[str] = meta.get("perturbation_list", [])

        if not b_datasets or not p_datasets:
            return ui.div(
                {"class": "empty-state"},
                ui.p("No datasets selected."),
            )

        def _cd_col_tooltip(p_db: str) -> str:
            try:
                preset_name = input.responsiveness_preset()
            except Exception:
                preset_name = DEFAULT_RESPONSIVENESS_PRESET
            thresh = get_responsiveness_label(preset_name, p_db)
            return (
                f"Responsive threshold: {thresh}. "
                "Click to view distributions for this perturbation dataset."
            )

        return build_topn_matrix_ui(
            binding_datasets=b_datasets,
            perturbation_datasets=p_datasets,
            topn_medians={},
            display_names=display_names,
            selected_binding=cd_selected_binding(),
            selected_perturbation=cd_selected_perturbation(),
            ns=session.ns,
            cell_slot=lambda b, p: ui.output_ui(f"cdcell_{b}__{p}"),
            col_tooltip=_cd_col_tooltip,
        )

    @output(suspend_when_hidden=False)
    @render.ui
    @_timed_render("cd_distribution_container")
    def cd_distribution_container() -> ui.Tag:
        """
        One box plot per pair in the selected row or column.

        Reads from all _cd_slot values to work during and after progressive loading.

        :trigger _cd_slot[*]: re-renders as perturbation slices land. :trigger
        cd_selected_binding: re-renders when a row is selected. :trigger
        cd_selected_perturbation: re-renders when a column is selected.

        """
        if (
            active_tab is not None
            and active_tab() != "Binding/Perturbation Comparisons"
        ):
            return ui.span()
        if _results_stale():
            return ui.span()
        meta = _run_meta_by_tab().get("Compare Datasets", {})
        if not meta:
            return ui.span()

        b_datasets: list[str] = meta.get("cd_binding_datasets", [])
        p_datasets: list[str] = meta.get("perturbation_list", [])

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

        # Collect available slot data (only perturbations that have landed).
        parts = [
            _cd_slot[p]()
            for p in p_datasets
            if p in _cd_slot and _cd_slot[p]() is not None and not _cd_slot[p]().empty
        ]
        if not parts:
            return ui.span()
        cd_data = pd.concat(parts, ignore_index=True)

        if b_sel is not None:
            pairs = [(b_sel, p_db) for p_db in p_datasets]
            x_col = "perturbation_source"
        else:
            pairs = [(b_db, p_sel) for b_db in b_datasets]
            x_col = "binding_label"

        fig = go.Figure()
        for b_db, p_db in pairs:
            sub = cd_data[
                (cd_data["binding_db"] == b_db) & (cd_data["perturbation_db"] == p_db)
            ]
            if sub.empty:
                continue
            mask = sub["percent_responsive"].notna()
            fig.add_trace(
                go.Box(
                    x=sub.loc[mask, x_col].values,
                    y=sub.loc[mask, "percent_responsive"].values,
                    name=sub.loc[mask, x_col].iloc[0] if mask.any() else "",
                    text=sub.loc[mask, "regulator_label"].values,
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
            return ui.div(
                {"class": "empty-state"},
                ui.p("No data for the selected datasets."),
            )

        fig.update_yaxes(title_text="% responsive in top N", range=[0, 100])
        fig.update_layout(margin=dict(l=50, r=20, t=40, b=80))
        return ui.div(
            {"style": "margin-top: 1.5rem;"},
            ui.HTML(to_html(fig, include_plotlyjs=False, full_html=False)),
        )

    # ===========================================================================
    # Tab 2: Compare Promoter Definitions
    # ===========================================================================

    # Pre-register one render per perturbation in the universe.
    def _make_cp_table_render(p_db: str) -> None:
        @output(id=f"cptable_{p_db}")
        @render.ui
        def _cptable() -> ui.Tag:
            """
            One perturbation's promoter comparison table, filled when its slice lands.

            :trigger _cp_slot[p_db]: re-renders only when this perturbation's data
            arrives.

            """
            df = _cp_slot[p_db]()
            if df is None:
                return ui.div(
                    {"class": "cp-table-loading"},
                    ui.p("Computing..."),
                )
            if df.empty:
                return ui.span()

            with reactive.isolate():
                meta = _run_meta_by_tab().get("Compare Promoter Definitions", {})
            included_ps: list[str] = meta.get("cp_included_promoter_sets", ["Kang"])
            pert_label = PERTURBATION_LABEL_MAP.get(p_db, p_db)
            with reactive.isolate():
                try:
                    preset_name = input.responsiveness_preset()
                except Exception:
                    preset_name = DEFAULT_RESPONSIVENESS_PRESET
            thresh_label = get_responsiveness_label(preset_name, p_db)

            binding_base_labels = [
                b for b in _BINDING_ORDER if b in df["binding_base_label"].unique()
            ]
            _th_style = "padding: 6px 10px; text-align: right;"

            header_cells = [
                ui.tags.th(
                    "Binding Dataset",
                    style="padding: 6px 10px; text-align: left;",
                ),
            ]
            for ps in included_ps:
                header_cells.append(ui.tags.th(ps, style=_th_style))

            data_rows: list[ui.Tag] = []
            for base_label in binding_base_labels:
                sub_b = df[df["binding_base_label"] == base_label]
                if sub_b.empty:
                    continue
                row_cells = [
                    ui.tags.td(
                        base_label,
                        style=(
                            "padding: 6px 10px; text-align: left; white-space: nowrap;"
                        ),
                    )
                ]
                for ps in included_ps:
                    sub_ps = sub_b[sub_b["promoter_set"] == ps]
                    if sub_ps.empty:
                        row_cells.append(
                            ui.tags.td(
                                "-", style="padding: 6px 10px; text-align: right;"
                            )
                        )
                    else:
                        per_reg = sub_ps.groupby("regulator_locus_tag")[
                            "percent_responsive"
                        ].median()
                        val = float(per_reg.median()) if not per_reg.empty else None
                        if val is not None and not pd.isna(val):
                            row_cells.append(
                                ui.tags.td(f"{val:.1f}%", style=_cell_style(val))
                            )
                        else:
                            row_cells.append(
                                ui.tags.td(
                                    "-", style="padding: 6px 10px; text-align: right;"
                                )
                            )
                data_rows.append(ui.tags.tr(*row_cells))

            if not data_rows:
                return ui.span()

            return ui.div(
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
                        ui.span(pert_label),
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

    for _p in _all_perturbation:
        _make_cp_table_render(_p)

    @output(suspend_when_hidden=False)
    @render.ui
    @_timed_render("cp_promoter_table")
    def cp_promoter_table() -> ui.Tag:
        """
        Scaffold: one ``output_ui`` slot per perturbation in the active run.

        Renders once per Execute from ``_run_meta``; each slot fills independently
        when its perturbation slice lands.

        :trigger _run_meta: re-renders when a new Execute starts.
        :trigger _results_stale: re-renders when results are invalidated.

        """
        if _results_stale():
            return ui.div(
                {"class": "empty-state"},
                ui.p("Click Execute Analysis to compute."),
            )
        meta = _run_meta_by_tab().get("Compare Promoter Definitions", {})
        if not meta:
            return ui.div(
                {"class": "empty-state"},
                ui.p("Click Execute Analysis to compute."),
            )
        pert_list: list[str] = meta.get("perturbation_list", [])
        if not pert_list:
            return ui.div(
                {"class": "empty-state"},
                ui.p("No perturbation datasets selected."),
            )

        slots = [ui.output_ui(f"cptable_{p}") for p in pert_list]
        return ui.div(
            {
                "style": (
                    "display: flex; flex-wrap: wrap;"
                    " gap: 1.5rem; margin-top: 0.5rem;"
                )
            },
            *slots,
        )

    # ===========================================================================
    # Tab 3: Compare Analysis Methods
    # ===========================================================================

    # Pre-register one render per perturbation in the universe.
    def _make_cm_table_render(p_db: str) -> None:
        @output(id=f"cmtable_{p_db}")
        @render.ui
        def _cmtable() -> ui.Tag:
            """
            One perturbation's method comparison table, filled when its slice lands.

            :trigger _cm_slot[p_db]: re-renders only when this perturbation's data
            arrives.

            """
            df = _cm_slot[p_db]()
            if df is None:
                return ui.div(
                    {"class": "cm-table-loading"},
                    ui.p("Computing..."),
                )
            if df.empty:
                return ui.span()

            pert_label = PERTURBATION_LABEL_MAP.get(p_db, p_db)
            with reactive.isolate():
                try:
                    preset_name = input.responsiveness_preset()
                except Exception:
                    preset_name = DEFAULT_RESPONSIVENESS_PRESET
            thresh_label = get_responsiveness_label(preset_name, p_db)
            variants_present = [
                v for v in SCORING_VARIANT_ORDER if v in df["scoring_variant"].unique()
            ]
            _th_style = "padding: 6px 10px; text-align: right;"

            header = ui.tags.tr(
                ui.tags.th(
                    "Scoring Variant", style="padding: 6px 10px; text-align: left;"
                ),
                ui.tags.th("Median % Responsive", style=_th_style),
            )

            data_rows: list[ui.Tag] = []
            for variant in variants_present:
                sub_v = df[df["scoring_variant"] == variant]
                if sub_v.empty:
                    continue
                per_reg = sub_v.groupby("regulator_locus_tag")[
                    "percent_responsive"
                ].median()
                val = float(per_reg.median()) if not per_reg.empty else None
                color = SCORING_VARIANT_COLORS.get(variant, "#888888")
                data_rows.append(
                    ui.tags.tr(
                        ui.tags.td(
                            ui.span(
                                {
                                    "style": (
                                        "display: inline-block; width: 10px;"
                                        " height: 10px; border-radius: 50%;"
                                        f" background: {color}; margin-right: 6px;"
                                    )
                                },
                            ),
                            variant,
                            style=(
                                "padding: 6px 10px; text-align: left;"
                                " white-space: nowrap;"
                            ),
                        ),
                        ui.tags.td(
                            (
                                f"{val:.1f}%"
                                if val is not None and not pd.isna(val)
                                else "-"
                            ),
                            style=(
                                _cell_style(val)
                                if val is not None and not pd.isna(val)
                                else "padding: 6px 10px; text-align: right;"
                            ),
                        ),
                    )
                )

            if not data_rows:
                return ui.span()

            return ui.div(
                {
                    "style": (
                        "flex: 1 1 0; border: 1px solid #ddd;"
                        " border-radius: 4px; overflow: hidden; min-width: 280px;"
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
                        ui.span(pert_label),
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
                    ui.tags.thead({"style": "background-color: #f5f5f5;"}, header),
                    ui.tags.tbody(*data_rows),
                ),
            )

    for _p in _all_perturbation:
        _make_cm_table_render(_p)

    @output(suspend_when_hidden=False)
    @render.ui
    @_timed_render("cm_method_table")
    def cm_method_table() -> ui.Tag:
        """
        Scaffold: one ``output_ui`` slot per perturbation in the active run.

        Renders once per Execute from ``_run_meta``; each slot fills independently
        when its perturbation slice lands.

        :trigger _run_meta: re-renders when a new Execute starts.
        :trigger _results_stale: re-renders when results are invalidated.

        """
        if _results_stale():
            return ui.div(
                {"class": "empty-state"},
                ui.p("Click Execute Analysis to compute."),
            )
        meta = _run_meta_by_tab().get("Compare Analysis Methods", {})
        if not meta:
            return ui.div(
                {"class": "empty-state"},
                ui.p("Click Execute Analysis to compute."),
            )
        pert_list: list[str] = meta.get("perturbation_list", [])
        if not pert_list:
            return ui.div(
                {"class": "empty-state"},
                ui.p("No perturbation datasets selected."),
            )

        slots = [ui.output_ui(f"cmtable_{p}") for p in pert_list]
        return ui.div(
            {
                "style": (
                    "display: flex; flex-wrap: wrap;"
                    " gap: 1.5rem; margin-top: 0.5rem;"
                )
            },
            *slots,
        )


__all__ = ["comparison_workspace_server"]
