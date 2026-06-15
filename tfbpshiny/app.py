from __future__ import annotations

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Any, Literal, cast

from dotenv import load_dotenv
from shiny import App, reactive, render, ui
from shiny.reactive import extended_task

from tfbpshiny.components import github_badge
from tfbpshiny.configure_logger import configure_logger
from tfbpshiny.modules.binding.server import binding_server
from tfbpshiny.modules.binding.ui import binding_ui
from tfbpshiny.modules.comparison.server import comparison_server
from tfbpshiny.modules.comparison.ui import comparison_ui
from tfbpshiny.modules.home.ui import HOME_CARD_NAV_TARGETS, home_ui
from tfbpshiny.modules.perturbation.server import perturbation_server
from tfbpshiny.modules.perturbation.ui import perturbation_ui
from tfbpshiny.modules.select_datasets.server import select_datasets_server
from tfbpshiny.modules.select_datasets.ui import selection_ui
from tfbpshiny.utils.vdb_init import check_local_cache, initialize_data

# Module UIs are declared once at startup. The actual output bindings inside
# each module only resolve after the server is registered (post-init), so the
# panels show Shiny's default blank/loading state until data is ready without
# any extra wrapper render functions.
_selection_ui = selection_ui("select_datasets")
_binding_ui = binding_ui("binding")
_perturbation_ui = perturbation_ui("perturbation")
_comparison_ui = comparison_ui("comparison")

if not os.getenv("DOCKER_ENV"):
    load_dotenv(dotenv_path=Path(".env"))

logger = logging.getLogger("shiny")

_log_dir = Path("tfbpshiny_log")
_log_dir.mkdir(exist_ok=True)
_log_file = str(_log_dir / f"tfbpshiny_{time.strftime('%Y%m%d-%H%M%S')}.log")
_log_level = int(os.getenv("TFBPSHINY_LOG_LEVEL", str(logging.INFO)))
_log_handler = cast(
    Literal["console", "file"], os.getenv("TFBPSHINY_LOG_HANDLER", "console")
)
configure_logger(
    "shiny", level=_log_level, handler_type=_log_handler, log_file=_log_file
)
configure_logger(
    "labretriever", level=_log_level, handler_type=_log_handler, log_file=_log_file
)

virtualdb_config: str = os.getenv(
    "VIRTUALDB_CONFIG",
    str(Path(__file__).parent / "brentlab_yeast_collection.yaml"),
)
hf_token: str | None = os.getenv("HF_TOKEN")


_not_ready_ui = ui.div(
    {
        "style": "display:flex; align-items:center; justify-content:center;"
        " height:60%; color:#888; text-align:center;"
    },
    ui.p("Please visit the Dataset selection tab first to load the data."),
)

app_ui = ui.page_navbar(
    ui.nav_panel("Home", home_ui()),
    ui.nav_panel("Dataset selection", ui.output_ui("selection_status"), _selection_ui),
    ui.nav_panel("Binding", ui.output_ui("binding_status"), _binding_ui),
    ui.nav_panel("Perturbation", ui.output_ui("perturbation_status"), _perturbation_ui),
    ui.nav_panel(
        "Binding/Perturbation Comparisons",
        ui.output_ui("comparison_status"),
        _comparison_ui,
    ),
    ui.nav_spacer(),
    ui.nav_control(github_badge()),
    title="TF Binding & Perturbation Explorer",
    id="main_nav",
    fillable=[
        "Dataset selection",
        "Binding",
        "Perturbation",
        "Binding/Perturbation Comparisons",
    ],
    navbar_options=ui.navbar_options(bg="#722F37", theme="dark"),
    header=ui.tags.head(
        ui.tags.script(src="plotly-3.5.0.min.js"),
        ui.include_css((Path(__file__).parent / "app.css").resolve()),
    ),
)


def app_server(input: Any, output: Any, session: Any) -> None:
    """Create shared reactive state and call all module servers."""

    @reactive.calc
    def _active_tab() -> str:
        return input.main_nav()

    # Navigate to the target tab when a home-page card title link is clicked.
    for _link_id, _target in HOME_CARD_NAV_TARGETS.items():

        def _make_nav_effect(link_id: str, target: str) -> None:
            @reactive.effect
            @reactive.event(getattr(input, link_id))
            def _nav_to_tab() -> None:
                ui.update_navset("main_nav", selected=target)

        _make_nav_effect(_link_id, _target)

    # Fires exactly once when init succeeds; registers all module servers.
    @reactive.effect
    def _register_modules() -> None:
        if _init_task.status() != "success":
            return

        vdb, app_datasets = _init_task.result()

        active_binding_datasets, active_perturbation_datasets, dataset_filters = (
            select_datasets_server(
                "select_datasets",
                vdb=vdb,
                app_datasets=app_datasets,
                logger=logger,
                active_tab=_active_tab,
                materialize_ready=materialize_ready,
            )
        )

        binding_server(
            "binding",
            active_binding_datasets=active_binding_datasets,
            dataset_filters=dataset_filters,
            vdb=vdb,
            app_datasets=app_datasets,
            logger=logger,
            active_tab=_active_tab,
            materialize_ready=materialize_ready,
        )

        perturbation_server(
            "perturbation",
            active_perturbation_datasets=active_perturbation_datasets,
            dataset_filters=dataset_filters,
            vdb=vdb,
            app_datasets=app_datasets,
            logger=logger,
            active_tab=_active_tab,
            materialize_ready=materialize_ready,
        )

        comparison_server(
            "comparison",
            active_binding_datasets=active_binding_datasets,
            active_perturbation_datasets=active_perturbation_datasets,
            dataset_filters=dataset_filters,
            vdb=vdb,
            logger=logger,
            active_tab=_active_tab,
            materialize_ready=materialize_ready,
        )

        # Start background materialization only after every module server has been
        # registered. Registration runs synchronously here and performs the only
        # registration-time DuckDB reads (e.g. get_regulator_display_name); invoking
        # the task last guarantees those reads complete before the materialize thread
        # touches the (non-thread-safe) connection.
        if not _materialize_started["done"]:
            _materialize_started["done"] = True
            _materialize_task.invoke(vdb)

    @extended_task
    async def _init_task(config: str, token: str | None) -> Any:
        """
        Run the fast VirtualDB initialization off the main thread.

        Materialization is deferred to ``_materialize_task`` so the app becomes
        interactive without waiting on the ~22-48s in-RAM copy of the data views.

        """
        missing = await asyncio.to_thread(check_local_cache, config)
        if missing:
            raise RuntimeError(
                "Data cache is insufficient. Contact administrator with "
                "an issue at https://github.com/BrentLab/tfbpshiny/issues. "
                f"Missing repos: {missing}"
            )
        return await asyncio.to_thread(
            initialize_data, config, token, defer_materialize=True
        )

    @extended_task
    async def _materialize_task(vdb: Any) -> bool:
        """
        Materialize the comparison data views into RAM off the main thread.

        Runs after the fast init completes. While it is running, every reactive
        that queries ``vdb`` is gated behind :func:`materialize_ready`, so this task
        is the sole user of the (non-thread-safe) DuckDB connection during the window.

        """
        from tfbpshiny.utils.vdb_materialize import materialize_comparison_views

        await asyncio.to_thread(materialize_comparison_views, vdb)
        return True

    def materialize_ready() -> bool:
        """Reactive predicate: True once background materialization has finished."""
        return _materialize_task.status() == "success"

    # Tracks whether background materialization has been kicked off (once per
    # session). Set inside _register_modules after all servers are registered.
    _materialize_started: dict[str, bool] = {"done": False}

    # Auto-start init on session load — no button required.
    _init_task.invoke(virtualdb_config, hf_token)

    _preparing_ui = ui.div(
        {"class": "pending-banner"},
        "Loading datasets. This can take up to ~10 seconds on a cold start. "
        "Thank you for your patience.",
    )

    # Shown on data-querying tabs after fast init while the background
    # materialization runs. Queries are gated until it completes, so this banner
    # tells the user the tab will unlock automatically.
    _optimizing_ui = ui.div(
        {"class": "pending-banner"},
        "Optimizing analysis data for faster queries. This can take up to ~50 seconds "
        "on a cold start; analysis tools unlock automatically when it completes.",
    )

    def _status_panel(ready_content: ui.Tag | None = None) -> ui.Tag:
        """Return a status message or empty span based on init task state."""
        status = _init_task.status()
        if status == "success":
            return ready_content if ready_content is not None else ui.span()
        if status in ("error", "cancelled"):
            err = _init_task.error() if status == "error" else None
            msg = str(err) if err else "Initialisation was cancelled."
            return ui.div(
                {
                    "style": "display:flex; align-items:center;"
                    " justify-content:center; padding:2rem; color:#b00;"
                    " text-align:center;"
                },
                ui.p(msg),
            )
        return _preparing_ui

    @render.ui
    def selection_status() -> ui.Tag:
        if _init_task.status() == "success" and not materialize_ready():
            return _optimizing_ui
        return _status_panel()

    @render.ui
    def binding_status() -> ui.Tag:
        if _init_task.status() == "success":
            return _optimizing_ui if not materialize_ready() else ui.span()
        return _status_panel(_not_ready_ui)

    @render.ui
    def perturbation_status() -> ui.Tag:
        if _init_task.status() == "success":
            return _optimizing_ui if not materialize_ready() else ui.span()
        return _status_panel(_not_ready_ui)

    @render.ui
    def comparison_status() -> ui.Tag:
        if _init_task.status() == "success":
            return _optimizing_ui if not materialize_ready() else ui.span()
        return _status_panel(_not_ready_ui)


app = App(
    ui=app_ui,
    server=app_server,
    static_assets=Path(__file__).parent / "www",
)
