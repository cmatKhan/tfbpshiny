from __future__ import annotations

import argparse
import os
import pathlib

from shiny import run_app

from tfbpshiny.configure_logger import LogLevel, configure_logger

_DEFAULT_DB_PATH = str(
    pathlib.Path(__file__).parent / "brentlab_yeast.duckdb"
)


def run_shiny(args: argparse.Namespace) -> None:
    """
    Start the Shiny app, pointing it at the pre-materialized DuckDB file.

    :param args: Parsed CLI arguments with ``db_path``, ``port``, ``host``, and
        ``debug`` attributes.

    """
    db_path = str(pathlib.Path(args.db_path).resolve())
    os.environ["TFBPSHINY_DB_PATH"] = db_path

    import logging

    log_level = LogLevel.from_string(args.log_level)
    os.environ["TFBPSHINY_LOG_LEVEL"] = str(log_level.value)
    os.environ["TFBPSHINY_LOG_HANDLER"] = args.log_handler
    configure_logger("shiny", level=log_level.value, handler_type=args.log_handler)
    logger = logging.getLogger("shiny")
    logger.info("Using DuckDB at: %s", db_path)

    kwargs: dict[str, object] = {"port": args.port, "host": args.host}
    if args.debug:
        kwargs.update({"reload": True, "reload_dirs": ["tfbpshiny"]})
    run_app("tfbpshiny.app:app", **kwargs)  # type: ignore


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tfbpshiny",
        description=(
            "tfbpshiny is a CLI with multiple utilities. Use --help after any command."
        ),
        epilog="Use 'tfbpshiny <command> --help' for more info on each command.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
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

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand: materialize (register if available)
    try:
        from tfbpshiny.materialize.cli import register_subparser as _register_materialize

        _register_materialize(subparsers)
    except ImportError:
        pass

    # Subcommand: shiny
    shiny_parser = subparsers.add_parser(
        "shiny",
        help="Start the Shiny app against a pre-materialized DuckDB file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    shiny_parser.add_argument(
        "--db-path",
        type=str,
        default=_DEFAULT_DB_PATH,
        help="Path to the pre-materialized brentlab_yeast.duckdb file.",
    )
    shiny_parser.add_argument(
        "--debug", action="store_true", help="Enable debug mode with auto-reload."
    )
    shiny_parser.add_argument(
        "--port", type=int, default=8000, help="Port to serve the Shiny app on."
    )
    shiny_parser.add_argument(
        "--host", type=str, default="127.0.0.1", help="Host to bind the Shiny app."
    )
    shiny_parser.set_defaults(func=run_shiny)

    return parser


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
