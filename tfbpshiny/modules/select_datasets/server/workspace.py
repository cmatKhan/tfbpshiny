"""Workspace server for the Select Datasets page."""

from __future__ import annotations

from collections.abc import Callable
from logging import Logger
from typing import Any

from labretriever import VirtualDB
from shiny import reactive, render, req, ui
from shiny.types import SilentException

from tfbpshiny.components import (
    matrix_cell,
    matrix_cell_button,
    matrix_header_cell,
    matrix_row_label,
    matrix_table,
    pending_regulator_banner,
)
from tfbpshiny.modules.select_datasets.queries import (
    matrix_cross_dataset_query,
    matrix_diagonal_query,
    regulator_breakdown_query,
    regulator_intersection_query,
)
from tfbpshiny.modules.select_datasets.ui import (
    diagonal_cell_modal_ui,
    off_diagonal_cell_modal_ui,
)
from tfbpshiny.utils.perf import perf, reset_render_counts
from tfbpshiny.utils.vdb_init import HIDDEN_FILTER_FIELDS


def select_datasets_workspace_server(
    input: Any,
    output: Any,
    session: Any,
    active_binding_datasets: reactive.Calc_[list[str]],
    active_perturbation_datasets: reactive.Calc_[list[str]],
    dataset_filters: reactive.Value[dict[str, Any]],
    pending_regulator_pair: reactive.Value[dict[str, Any] | None],
    vdb: VirtualDB,
    logger: Logger,
    active_tab: reactive.Value[str] | None = None,
    materialize_ready: Callable[[], bool] | None = None,
) -> None:
    """Render the sample-count matrix for all active datasets."""

    session.on_flush(lambda: reset_render_counts(session.id))

    display_names: dict[str, str] = {
        db_name: vdb.get_tags(db_name).get("display_name", db_name)
        for db_name in vdb.get_datasets()
    }

    # Build a db_name -> DOI URL map from the config and datacards.
    # Two db_names may share a repo (e.g. chec_m2025 and degron both come from
    # BrentLab/mahendrawada_2025), so they share the same DOI.
    _doi_map: dict[str, str] = {}
    for _repo_id, _repo_cfg in vdb.config.repositories.items():
        if not _repo_cfg.dataset:
            continue
        for _ds_cfg in _repo_cfg.dataset.values():
            _db = getattr(_ds_cfg, "db_name", None)
            if not _db:
                continue
            _dc = vdb.datacards.get(_repo_id)
            if _dc:
                _doi = _dc.dataset_card.doi
                if _doi:
                    _doi_map[_db] = _doi

    def _row_label(db_name: str) -> str | ui.Tag:
        name = display_names.get(db_name, db_name)
        doi = _doi_map.get(db_name)
        if doi:
            return ui.tags.a(name, href=doi, target="_blank")
        return name

    # Canonical dataset ordering from vdb — used for button IDs and cross_dataset keys
    # so that all lookups are consistent regardless of active-list ordering.
    all_dbs: list[str] = list(vdb.get_datasets())
    _dbs_rank: dict[str, int] = {db: i for i, db in enumerate(all_dbs)}

    # Stable dataset list — updated only when the active dataset set actually changes.
    _settled_val: reactive.Value[list[str]] = reactive.value([])

    @reactive.effect
    def _sync_settled_datasets() -> None:
        """
        Write the combined active dataset list to ``_settled_val`` only when it changes.

        :trigger active_binding_datasets: re-fires when binding selection changes.
        :trigger active_perturbation_datasets: re-fires when perturbation selection
        changes. :trigger active_tab: silently blocks when another tab is active.

        """
        if active_tab is not None:
            req(active_tab() == "Dataset selection")
        new = active_binding_datasets() + active_perturbation_datasets()
        with reactive.isolate():
            if new != _settled_val():
                _settled_val.set(new)

    @reactive.calc
    def _matrix_data() -> dict[str, Any]:
        """
        Compute per-dataset regulator/sample counts and pairwise common-regulator
        counts.

        :trigger: ``_settled_val`` — re-runs when the active dataset list actually
            changes; only invalidated when list content changes, so returning to this
            tab without changing datasets hits the cache.
            ``dataset_filters`` — re-runs when any filter changes.
        :returns: Dict with keys ``"diagonal"`` — ``{db_name: {"regulators": int,
            "samples": int}}``; ``"cross_dataset"`` — ``{(db_i, db_j):
            {"common_regulators": int, "samples_a": int, "samples_b": int}}``.

        """
        # Gate vdb access until background materialization finishes; the DuckDB
        # connection is not safe to touch while materialization mutates it.
        if materialize_ready is not None:
            req(materialize_ready())
        with perf(session.id, "select_datasets.workspace", "_matrix_data"):
            active = _settled_val()
            filters = dataset_filters()

            diagonal: dict[str, dict[str, int]] = {}
            cross_dataset: dict[tuple[str, str], dict[str, int]] = {}

            if not active:
                return {"diagonal": diagonal, "cross_dataset": cross_dataset}

            # One query for all diagonal counts.
            diag_sql, diag_params = matrix_diagonal_query(active, filters)
            diag_df = vdb.query(diag_sql, **diag_params)
            for _, row in diag_df.iterrows():
                diagonal[str(row["db_name"])] = {
                    "regulators": int(row["n_regulators"]),
                    "samples": int(row["n_samples"]),
                }

            # One query for all cross-dataset pair counts.
            pairs = [
                (active[i], active[j])
                for i in range(len(active))
                for j in range(i + 1, len(active))
            ]
            if pairs:
                cross_sql, cross_params = matrix_cross_dataset_query(pairs, filters)
                cross_df = vdb.query(cross_sql, **cross_params)
                for _, row in cross_df.iterrows():
                    db_a_raw, db_b_raw = str(row["pair_id"]).split("__", 1)
                    n_common = int(row["n_common"])
                    s_a = int(row["samples_a"])
                    s_b = int(row["samples_b"])
                    # Normalize to canonical all_dbs order so both _on_click (which
                    # uses canonical order) and matrix_content (via canonical_pair)
                    # always hit the correct key.
                    if _dbs_rank.get(db_a_raw, -1) <= _dbs_rank.get(db_b_raw, -1):
                        cross_dataset[(db_a_raw, db_b_raw)] = {
                            "common_regulators": n_common,
                            "samples_a": s_a,
                            "samples_b": s_b,
                        }
                    else:
                        cross_dataset[(db_b_raw, db_a_raw)] = {
                            "common_regulators": n_common,
                            "samples_a": s_b,
                            "samples_b": s_a,
                        }

            return {"diagonal": diagonal, "cross_dataset": cross_dataset}

    @reactive.calc
    def _committed_regulator_pair() -> tuple[str, str] | None:
        """
        Derive the committed regulator-filter pair from ``dataset_filters``.

        Reads the ``from_pair_db`` key stored by ``_apply_pending`` when a
        pairwise filter is committed. Returns ``None`` when no pairwise filter
        is active.

        :trigger: ``dataset_filters`` — re-runs on every filter change.

        """
        for ds_filters in dataset_filters().values():
            reg = ds_filters.get("regulator_locus_tag")
            if reg and reg.get("from_pair_db"):
                pair = reg["from_pair_db"]
                return (str(pair[0]), str(pair[1]))
        return None

    # Tracks which off-diagonal modal is currently open, so that the single
    # _on_queue_common_regulators effect knows which pair to compute.
    _open_modal_pair: reactive.Value[tuple[str, str] | None] = reactive.value(None)

    def _make_diagonal_effect(db_name: str) -> None:
        """
        Create the click effect for one diagonal cell.

        :param db_name: Dataset identifier; used to derive the button input ID and to
            query the breakdown data.

        """
        btn_id = f"diag_{db_name}"

        @reactive.effect
        @reactive.event(input[btn_id])
        def _on_click() -> None:
            """
            Compute regulator/sample multiplicity for this dataset and show the diagonal
            cell modal.

            :trigger: ``input[diag_{db_name}]`` — fires when the user clicks the
                diagonal matrix cell button for this dataset.

            """
            if materialize_ready is not None:
                req(materialize_ready())
            with perf(session.id, "select_datasets.workspace", "diagonal._on_click"):
                filters = dataset_filters().get(db_name)

                all_cols = vdb.get_fields(f"{db_name}_meta")
                remove_cols = (
                    {"sample_id"}
                    | {c for c in all_cols if c.lower().startswith("regulator")}
                    | HIDDEN_FILTER_FIELDS.get("*", set())
                    | HIDDEN_FILTER_FIELDS.get(db_name, set())
                )
                candidate_cols = [c for c in all_cols if c not in remove_cols]

                sql, params = regulator_breakdown_query(
                    db_name, candidate_cols, filters
                )
                row = vdb.query(sql, **params).iloc[0]
                n_multi = int(row["n_multi"])

                if n_multi == 0:
                    multi_regulator_sample_breakdown: dict = {"uniform": True}
                else:
                    diff_cols = [c for c in candidate_cols if row[c] > 0]
                    multi_regulator_sample_breakdown = {
                        "uniform": False,
                        "n_multi": n_multi,
                        "differentiating_columns": diff_cols,
                    }

                display_name = display_names.get(db_name, db_name)
                ui.modal_show(
                    diagonal_cell_modal_ui(
                        display_name, multi_regulator_sample_breakdown
                    )
                )

    def _make_off_diagonal_effect(db_a: str, db_b: str) -> None:
        """Register the click effect for one off-diagonal cell."""
        btn_id = f"offdiag_{db_a}__{db_b}"

        @reactive.effect
        @reactive.event(input[btn_id])
        def _on_click() -> None:
            """
            If this pair is the committed regulator filter, clear the filter. Otherwise
            show the off-diagonal cell modal.

            :trigger: ``input[offdiag_{db_a}__{db_b}]`` — fires when the user
                clicks the off-diagonal matrix cell button for this pair.

            """
            with perf(
                session.id, "select_datasets.workspace", "off_diagonal._on_click"
            ):
                if _committed_regulator_pair() == (db_a, db_b):
                    # Clear the committed pairwise regulator filter from all datasets.
                    current = dict(dataset_filters())
                    for db_name in list(current):
                        ds_filters = dict(current[db_name])
                        ds_filters.pop("regulator_locus_tag", None)
                        if ds_filters:
                            current[db_name] = ds_filters
                        else:
                            current.pop(db_name)
                    dataset_filters.set(current)
                    return
                data = _matrix_data()
                info = data["cross_dataset"].get((db_a, db_b), {})
                n_common = info.get("common_regulators", 0)
                # Check whether a different pair is already pending.
                existing = pending_regulator_pair()
                pending_display = None
                if existing is not None and existing["pair"] != (db_a, db_b):
                    pending_display = existing["display"]
                _open_modal_pair.set((db_a, db_b))
                ui.modal_show(
                    off_diagonal_cell_modal_ui(
                        display_names.get(db_a, db_a),
                        display_names.get(db_b, db_b),
                        n_common,
                        pending_pair_display=pending_display,
                    )
                )

    # Single effect that responds to the "Select common regulators" button
    # across all off-diagonal modals. Uses _open_modal_pair to determine which
    # pair was active when the modal was opened.
    @reactive.effect
    @reactive.event(input.modal_queue_common_regulators)
    def _on_queue_common_regulators() -> None:
        """
        Compute the regulator intersection eagerly and store it as a pending regulator
        filter. The filter is committed to ``dataset_filters`` only when the user clicks
        Apply in the sidebar.

        :trigger: ``input.modal_queue_common_regulators`` — fires when the user
            clicks the "Select common regulators" button in the off-diagonal modal.

        """
        if materialize_ready is not None:
            req(materialize_ready())
        with perf(
            session.id,
            "select_datasets.workspace",
            "_on_queue_common_regulators",
        ):
            pair = _open_modal_pair()
            if pair is None:
                return
            db_a, db_b = pair
            # Exclude any existing regulator_locus_tag filter so the pairwise
            # intersection is computed from the full regulator set for each
            # dataset (subject to other filters only).
            filters = dataset_filters()
            fa = {
                k: v
                for k, v in (filters.get(db_a) or {}).items()
                if k != "regulator_locus_tag"
            } or None
            fb = {
                k: v
                for k, v in (filters.get(db_b) or {}).items()
                if k != "regulator_locus_tag"
            } or None
            sql, params = regulator_intersection_query(db_a, db_b, fa, fb)
            df = vdb.query(sql, **params)
            common_tags = df["regulator_locus_tag"].dropna().astype(str).tolist()
            if not common_tags:
                ui.modal_remove()
                _open_modal_pair.set(None)
                return
            pending_regulator_pair.set(
                {
                    "pair": (db_a, db_b),
                    "common_tags": common_tags,
                    "display": (
                        display_names.get(db_a, db_a),
                        display_names.get(db_b, db_b),
                    ),
                }
            )
            _open_modal_pair.set(None)
            ui.modal_remove()

    @reactive.effect
    @reactive.event(input.cancel_pending_regulator)
    def _on_cancel_pending() -> None:
        """
        Clear the pending regulator filter without committing it.

        :trigger: ``input.cancel_pending_regulator`` — fires when the user
            clicks the Cancel button in the pending filter banner.

        """
        pending_regulator_pair.set(None)

    # Register click effects at startup for every dataset and every ordered
    # pair that vdb knows about. Inactive datasets have no buttons rendered,
    # so their effects are registered but never triggered.
    for _db in all_dbs:
        _make_diagonal_effect(_db)
    for _i, _db_a in enumerate(all_dbs):
        for _db_b in all_dbs[_i + 1 :]:
            _make_off_diagonal_effect(_db_a, _db_b)

    @render.ui
    def matrix_content() -> ui.Tag:
        active = _settled_val()

        if not active:
            return ui.card(
                ui.card_body(
                    ui.p(
                        "Select datasets from the sidebar to view sample counts.",
                        class_="text-muted",
                    )
                )
            )

        try:
            data = _matrix_data()
        except SilentException:
            raise
        except Exception:
            logger.exception("Failed to compute matrix data")
            return ui.card(
                ui.card_body(
                    ui.p(
                        "Failed to load dataset matrix. Check that filters are valid.",
                        class_="text-danger",
                    )
                )
            )
        diagonal = data["diagonal"]
        cross_dataset = data["cross_dataset"]

        pending = pending_regulator_pair()
        committed = _committed_regulator_pair()

        # Pending banner shown above the matrix when a regulator filter is queued.
        banner_tags: list[ui.Tag] = []
        if pending is not None:
            display_a, display_b = pending["display"]
            n_pending = len(pending["common_tags"])
            banner_tags = [pending_regulator_banner(display_a, display_b, n_pending)]

        # --- header row ---
        header_cells = [matrix_header_cell("Dataset", row=True)]
        for db_name in active:
            header_cells.append(matrix_header_cell(display_names.get(db_name, db_name)))

        # --- body rows ---
        body_rows: list[ui.Tag] = []
        for row_i, db_row in enumerate(active):
            cells: list[ui.Tag] = [matrix_row_label(_row_label(db_row))]

            for col_i, db_col in enumerate(active):
                if col_i < row_i:
                    # lower triangle — empty
                    cells.append(matrix_cell("empty"))
                    continue

                if col_i == row_i:
                    # diagonal — regulator count + sample count
                    info = diagonal.get(db_row, {})
                    cells.append(
                        matrix_cell(
                            "diagonal",
                            matrix_cell_button(
                                session.ns(f"diag_{db_row}"),
                                f"{info.get('regulators', 0):,} regulators / "
                                f"{info.get('samples', 0):,} samples",
                            ),
                        )
                    )
                else:
                    # upper triangle — common regulators only
                    # Derive canonical pair (matches effect registration order and
                    # cross_dataset key normalization in _matrix_data).
                    if _dbs_rank.get(db_row, -1) < _dbs_rank.get(db_col, -1):
                        canonical_pair: tuple[str, str] = (db_row, db_col)
                    else:
                        canonical_pair = (db_col, db_row)
                    info = cross_dataset.get(canonical_pair, {})
                    offdiag_btn_id = f"offdiag_{canonical_pair[0]}__{canonical_pair[1]}"
                    is_active = committed == canonical_pair
                    is_pending = (
                        pending is not None and pending["pair"] == canonical_pair
                    )
                    tooltip_text: str | None = None
                    if is_active:
                        tooltip_text = "Click to remove the regulator filter"
                    elif is_pending:
                        tooltip_text = (
                            "Regulator filter pending — click Apply to commit"
                        )
                    cells.append(
                        matrix_cell(
                            "interactive",
                            matrix_cell_button(
                                session.ns(offdiag_btn_id),
                                f"{info.get('common_regulators', 0):,} "
                                "common regulators",
                                tooltip=tooltip_text,
                            ),
                            active=is_active,
                            pending=is_pending,
                        )
                    )

            body_rows.append(ui.tags.tr(*cells))

        return ui.div(
            *banner_tags,
            matrix_table(ui.tags.tr(*header_cells), *body_rows),
        )


__all__ = ["select_datasets_workspace_server"]
