"""
CLI handler for the ``tfbpshiny materialize`` subcommand.

Registers the subparser and implements ``run_materialize``, which loads
VirtualDB from the YAML config and calls :func:`~.coordinator.materialize`.
"""

from __future__ import annotations

import argparse
import logging
import os
import pathlib
import sys
from typing import Literal, cast

from tfbpshiny.configure_logger import LogLevel, configure_logger


def run_materialize(args: argparse.Namespace) -> None:
    """
    Entry point for the ``tfbpshiny materialize`` subcommand.

    Downloads or opens the VirtualDB from the YAML config, then runs the full
    materialization pipeline, writing the output ``.duckdb`` file.

    :param args: Parsed CLI namespace from :func:`register_subparser`.

    """
    from labretriever import VirtualDB

    from tfbpshiny.materialize.coordinator import materialize

    log_level = LogLevel.from_string(args.log_level)
    configure_logger(
        "shiny",
        level=log_level.value,
        handler_type=cast(Literal["console", "file"], args.log_handler),
    )
    logger = logging.getLogger("shiny")

    hf_token: str | None = args.token or os.getenv("HF_TOKEN")

    output_path = pathlib.Path(args.output)
    if output_path.exists():
        try:
            answer = input(
                f"\n'{output_path}' already exists. Overwrite? [y/N] "
            ).strip().lower()
        except EOFError:
            answer = ""
        if answer in ("y", "yes"):
            output_path.unlink()
            logger.info("Deleted existing file: %s", output_path)
        else:
            print(
                f"Aborted. Rename or move '{output_path}' out of the current "
                "directory and re-run to create a fresh database."
            )
            sys.exit(0)

    logger.info("Initializing VirtualDB from %s …", args.config)
    try:
        vdb = VirtualDB(args.config, token=hf_token, local_files_only=False)
    except Exception:
        logger.exception("Failed to initialize VirtualDB.")
        sys.exit(1)

    logger.info(
        "Writing materialized database to %s …", args.output
    )
    try:
        materialize(args.output, vdb, args)
    except Exception:
        logger.exception("Materialization failed.")
        sys.exit(1)


def register_subparser(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """
    Register the ``materialize`` subcommand on the parent parser's subparsers.

    :param subparsers: The ``subparsers`` action from the parent
        ``argparse.ArgumentParser``.

    """
    p = subparsers.add_parser(
        "materialize",
        help=(
            "Materialize all dataset views into a persistent DuckDB file.  "
            "Run once offline; the resulting .duckdb file is used by the app."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the VirtualDB YAML configuration file.",
    )
    p.add_argument(
        "--output",
        type=str,
        default="brentlab_yeast.duckdb",
        help="Output path for the materialized .duckdb file.",
    )
    p.add_argument(
        "--methods",
        type=str,
        default="pearson,spearman",
        help="Comma-separated correlation methods to compute.",
    )
    p.add_argument(
        "--top-n",
        dest="top_n_values",
        type=int,
        action="append",
        default=None,
        metavar="N",
        help=(
            "Top-N cutoff for the top-N responsive-ratio analysis "
            "(repeatable; e.g. --top-n 25 --top-n 50)."
        ),
    )
    p.add_argument(
        "--effect-threshold",
        dest="effect_thresholds",
        type=float,
        action="append",
        default=None,
        metavar="THRESHOLD",
        help=(
            "Effect-size threshold for responsiveness "
            "(repeatable; e.g. --effect-threshold 0.0 --effect-threshold 1.0)."
        ),
    )
    p.add_argument(
        "--pvalue-threshold",
        dest="pvalue_thresholds",
        type=float,
        action="append",
        default=None,
        metavar="THRESHOLD",
        help=(
            "Adjusted p-value threshold for responsiveness "
            "(repeatable; e.g. --pvalue-threshold 0.05 --pvalue-threshold 0.1)."
        ),
    )
    p.add_argument(
        "--skip-correlations",
        action="store_true",
        default=False,
        help="Skip the pairwise correlation computation.",
    )
    p.add_argument(
        "--skip-topn",
        action="store_true",
        default=False,
        help="Skip the top-N responsive-ratio computation.",
    )
    p.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace token for private repos (falls back to HF_TOKEN env var).",
    )
    p.set_defaults(func=_run_with_defaults)


def _run_with_defaults(args: argparse.Namespace) -> None:
    """Apply list-argument defaults then delegate to :func:`run_materialize`."""
    if args.top_n_values is None:
        args.top_n_values = [25]
    if args.effect_thresholds is None:
        args.effect_thresholds = [0.0]
    if args.pvalue_thresholds is None:
        args.pvalue_thresholds = [0.05]
    run_materialize(args)
