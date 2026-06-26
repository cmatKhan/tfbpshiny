from __future__ import annotations

import argparse
import logging
import os
import pathlib
import sys
from typing import Literal, cast

from shiny import run_app

from tfbpshiny.configure_logger import LogLevel, configure_logger

_DEFAULT_VIRTUALDB_CONFIG = str(
    __import__("pathlib").Path(__file__).parent / "brentlab_yeast_collection.yaml"
)

_DEFAULT_CACHE_DIR = "./tfbpshiny_hf_cache"


def _apply_cache_dir(cache_dir: str) -> None:
    """
    Set ``HF_CACHE_DIR`` to the resolved absolute path before any HF imports.

    Must be called before importing labretriever or huggingface_hub so that
    ``snapshot_download`` and ``VirtualDB`` see the overridden path.

    :param cache_dir: Path to the HuggingFace cache directory.

    """
    os.environ["HF_CACHE_DIR"] = str(pathlib.Path(cache_dir).resolve())


def _run_initialize(
    virtualdb_config: str,
    hf_token: str | None,
    log_level: int,
    log_handler: Literal["console", "file"],
) -> None:
    """
    Download all dataset files into the local HuggingFace cache and verify views.

    Exits the process with code 1 if any download or view-verification step fails.

    :param virtualdb_config: Path to the VirtualDB YAML config file.
    :param hf_token: Optional HuggingFace token for private repo access.
    :param log_level: Numeric logging level (e.g. ``logging.INFO``).
    :param log_handler: Handler type passed to :func:`configure_logger`.

    """
    from tfbpshiny.utils.vdb_init import initialize_data

    configure_logger("shiny", level=log_level, handler_type=log_handler)
    logger = logging.getLogger("shiny")

    cache_msg = os.environ.get("HF_CACHE_DIR", "(huggingface default)")
    logger.info("Downloading all datasets into HuggingFace cache: %s", cache_msg)
    try:
        vdb, _ = initialize_data(virtualdb_config, hf_token, local_files_only=False)
    except Exception:
        logger.exception("Cache initialization failed.")
        sys.exit(1)

    logger.info("Verifying all dataset views are readable...")
    views_df = vdb.query(
        "SELECT view_name FROM duckdb_views()"
        " WHERE schema_name = 'main'"
        "   AND view_name NOT LIKE 'duckdb_%'"
        "   AND view_name NOT LIKE 'sqlite_%'"
        "   AND view_name NOT LIKE 'pragma_%'"
        " ORDER BY view_name"
    )
    view_names = views_df["view_name"].tolist()
    failed: list[str] = []
    for view_name in view_names:
        try:
            df = vdb.query(f'SELECT * FROM "{view_name}" LIMIT 1')
            logger.info("  OK  %-30s  (%d col(s))", view_name, len(df.columns))
        except Exception:
            logger.exception("  FAIL %s", view_name)
            failed.append(view_name)

    if failed:
        logger.error("Verification failed for: %s", ", ".join(failed))
        sys.exit(1)

    logger.info("Cache initialization complete.")


def run_launch(args: argparse.Namespace) -> None:
    """
    Download the dataset cache (unless ``--skip-initialize`` is set), then start the
    app.

    By default uses ``./tfbpshiny_hf_cache`` as the HuggingFace cache directory so
    a plain ``python -m tfbpshiny launch`` is self-contained: it downloads data on
    first run and serves it on subsequent runs from the same local directory.

    """
    cache_dir: str = args.cache_dir
    _apply_cache_dir(cache_dir)

    log_level = LogLevel.from_string(args.log_level)
    hf_token: str | None = os.getenv("HF_TOKEN")

    if not args.skip_initialize:
        _run_initialize(
            virtualdb_config=args.virtualdb_config,
            hf_token=hf_token,
            log_level=log_level.value,
            log_handler=cast(Literal["console", "file"], args.log_handler),
        )

    # Env vars are the only reliable way to pass config to uvicorn reload workers,
    # which re-import app.py in a subprocess and cannot see in-process mutations.
    os.environ["TFBPSHINY_LOG_LEVEL"] = str(log_level.value)
    os.environ["TFBPSHINY_LOG_HANDLER"] = args.log_handler
    os.environ["VIRTUALDB_CONFIG"] = args.virtualdb_config
    os.environ["TFBPSHINY_MATERIALIZE"] = "0" if args.no_materialize else "1"

    kwargs: dict[str, object] = {"port": args.port, "host": args.host}
    if args.debug:
        kwargs.update({"reload": True, "reload_dirs": ["tfbpshiny/shiny_app"]})
    run_app("tfbpshiny.app:app", **kwargs)  # type: ignore


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tfbpshiny",
        description=(
            "tfbpshiny — TF Binding and Perturbation Explorer."
            " Use --help after any command."
        ),
        epilog="Use 'tfbpshiny <command> --help' for more info on each command.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set logging level.",
    )
    parser.add_argument(
        "--log-handler",
        type=str,
        default="console",
        choices=["console", "file"],
        help="Set log handler type.",
    )
    parser.add_argument(
        "--virtualdb-config",
        type=str,
        default=_DEFAULT_VIRTUALDB_CONFIG,
        help="Path to the VirtualDB YAML configuration file.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    from tfbpshiny.materialize.cli import register_subparser as _register_materialize

    _register_materialize(subparsers)

    launch_parser = subparsers.add_parser(
        "launch",
        help="Download the dataset cache (first run) and start the Shiny app.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    launch_parser.add_argument(
        "--cache-dir",
        type=str,
        default=_DEFAULT_CACHE_DIR,
        help=(
            "HuggingFace cache directory. Datasets are downloaded here on first run "
            "and read from here on subsequent runs. "
            "Equivalent to setting HF_CACHE_DIR."
        ),
    )
    launch_parser.add_argument(
        "--skip-initialize",
        action="store_true",
        default=False,
        help=(
            "Skip the dataset download and verification step. "
            "Use when the cache is already populated and you want a faster startup."
        ),
    )
    launch_parser.add_argument(
        "--no-materialize",
        action="store_true",
        default=False,
        help=(
            "Disable in-memory materialization of dataset views at startup. "
            "Reduces startup memory at the cost of slower query performance. "
            "Equivalent to setting TFBPSHINY_MATERIALIZE=0."
        ),
    )
    launch_parser.add_argument(
        "--port", type=int, default=8000, help="Port to serve the Shiny app on."
    )
    launch_parser.add_argument(
        "--host", type=str, default="127.0.0.1", help="Host to bind the Shiny app."
    )
    launch_parser.add_argument(
        "--debug", action="store_true", help="Enable debug mode with auto-reload."
    )
    launch_parser.set_defaults(func=run_launch)

    return parser


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
