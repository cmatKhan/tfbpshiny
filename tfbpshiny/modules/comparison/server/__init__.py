from __future__ import annotations

from collections.abc import Callable
from logging import Logger
from typing import Any

from labretriever import VirtualDB
from shiny import module, reactive

from tfbpshiny.modules.comparison.server.workspace import comparison_workspace_server


@module.server
def comparison_server(
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
    """Combined sidebar + workspace server for the Comparison module."""
    comparison_workspace_server(
        input,
        output,
        session,
        active_binding_datasets=active_binding_datasets,
        active_perturbation_datasets=active_perturbation_datasets,
        dataset_filters=dataset_filters,
        vdb=vdb,
        logger=logger,
        active_tab=active_tab,
        materialize_ready=materialize_ready,
    )


__all__ = ["comparison_server"]
