from __future__ import annotations

from logging import Logger
from typing import Any

import duckdb
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
    conn: duckdb.DuckDBPyConnection,
    app_datasets: AppDatasets,
    logger: Logger,
    active_tab: reactive.Calc_[str] | None = None,
) -> tuple[
    reactive.Calc_[list[str]],
    reactive.Calc_[list[str]],
    reactive.Value[dict[str, Any]],
]:
    """Combined sidebar + workspace server for the Select Datasets module."""
    active_binding_datasets, active_perturbation_datasets, dataset_filters = (
        select_datasets_sidebar_server(
            input,
            output,
            session,
            conn=conn,
            app_datasets=app_datasets,
            logger=logger,
        )
    )
    select_datasets_workspace_server(
        input,
        output,
        session,
        active_binding_datasets=active_binding_datasets,
        active_perturbation_datasets=active_perturbation_datasets,
        dataset_filters=dataset_filters,
        conn=conn,
        logger=logger,
    )
    return active_binding_datasets, active_perturbation_datasets, dataset_filters


__all__ = ["select_datasets_server"]
