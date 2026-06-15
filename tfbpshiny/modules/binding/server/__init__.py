from __future__ import annotations

from collections.abc import Callable
from logging import Logger
from typing import Any

from labretriever import VirtualDB
from shiny import module, reactive

from tfbpshiny.modules.binding.server.workspace import binding_workspace_server
from tfbpshiny.utils.vdb_init import AppDatasets


@module.server
def binding_server(
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
    """Combined sidebar + workspace server for the Binding module."""
    binding_workspace_server(
        input,
        output,
        session,
        active_binding_datasets=active_binding_datasets,
        dataset_filters=dataset_filters,
        vdb=vdb,
        app_datasets=app_datasets,
        logger=logger,
        active_tab=active_tab,
        materialize_ready=materialize_ready,
    )


__all__ = ["binding_server"]
