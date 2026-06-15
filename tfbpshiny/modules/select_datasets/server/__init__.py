from __future__ import annotations

from collections.abc import Callable
from logging import Logger
from typing import Any

from labretriever import VirtualDB
from shiny import module, reactive

from tfbpshiny.modules.select_datasets.server.sidebar import (
    select_datasets_sidebar_server,
)
from tfbpshiny.modules.select_datasets.server.workspace import (
    select_datasets_workspace_server,
)
from tfbpshiny.utils.vdb_init import AppDatasets


@module.server
def select_datasets_server(
    input: Any,
    output: Any,
    session: Any,
    vdb: VirtualDB,
    app_datasets: AppDatasets,
    logger: Logger,
    active_tab: reactive.Calc_[str] | None = None,
    materialize_ready: Callable[[], bool] | None = None,
) -> tuple[
    reactive.Calc_[list[str]],
    reactive.Calc_[list[str]],
    reactive.Value[dict[str, Any]],
]:
    """Combined sidebar + workspace server for the Select Datasets module."""

    # Pending pairwise regulator filter: set by workspace when the user clicks
    # "Select common regulators" in the off-diagonal modal; committed to
    # dataset_filters by sidebar when the user clicks Apply.
    _pending_regulator_pair: reactive.Value[dict[str, Any] | None] = reactive.value(
        None
    )

    active_binding_datasets, active_perturbation_datasets, dataset_filters = (
        select_datasets_sidebar_server(
            input,
            output,
            session,
            vdb=vdb,
            app_datasets=app_datasets,
            logger=logger,
            active_tab=active_tab,
            pending_regulator_pair=_pending_regulator_pair,
            materialize_ready=materialize_ready,
        )
    )
    select_datasets_workspace_server(
        input,
        output,
        session,
        active_binding_datasets=active_binding_datasets,
        active_perturbation_datasets=active_perturbation_datasets,
        dataset_filters=dataset_filters,
        pending_regulator_pair=_pending_regulator_pair,
        vdb=vdb,
        logger=logger,
        active_tab=active_tab,
        materialize_ready=materialize_ready,
    )
    return active_binding_datasets, active_perturbation_datasets, dataset_filters


__all__ = ["select_datasets_server"]
