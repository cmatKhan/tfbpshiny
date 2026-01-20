import logging
import os
import time
from pathlib import Path
from typing import Literal, cast

from dotenv import load_dotenv
from shiny import App, ui

from configure_logger import configure_logger

# Only load .env if not running in production
if not os.getenv("DOCKER_ENV"):
    load_dotenv(dotenv_path=Path(".env"))

logger = logging.getLogger("shiny")

# configure the logger
log_file = f"tfbpshiny_{time.strftime('%Y%m%d-%H%M%S')}.log"
log_level = int(os.getenv("TFBPSHINY_LOG_LEVEL", "10"))
handler_type = cast(
    Literal["console", "file"], os.getenv("TFBPSHINY_LOG_HANDLER", "console")
)
configure_logger(
    "shiny",
    level=log_level,
    handler_type=handler_type,
    log_file=log_file,
)

app_ui = ui.page_fillable(
    ui.panel_title(
        "TF Binding and Perturbation", window_title="TF Binding and Perturbation"
    ),
    ui.include_css((Path(__file__).parent / "style.css").resolve()),
)


def app_server(input, output, session):
    pass


# Create an app instance
app = App(ui=app_ui, server=app_server)
