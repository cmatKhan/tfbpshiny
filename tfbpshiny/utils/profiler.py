"""Timing context manager for the profiler logger."""

from __future__ import annotations

import time
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime, timezone
from logging import Logger


@contextmanager
def profile_span(
    logger: Logger,
    op: str,
    module: str = "",
    dataset: str = "",
    context: str = "",
) -> Generator[None]:
    """
    Time a block of code and emit one fixed-schema pipe-delimited record to *logger*.

    The record always has exactly seven pipe-separated columns so it can be read
    without regex using ``pandas.read_csv(sep="|")``:

    ::

        PROFILE | timestamp | elapsed_s | op | module | dataset | context

    :param logger: The ``"profiler"`` logger.
    :param op: Operation label, e.g. ``"vdb.query"`` or ``"plot.build"``.
    :param module: Shiny module name, e.g. ``"binding"``. Empty for init-time spans.
    :param dataset: Dataset(s) involved, e.g. ``"harbison"`` or ``"harbison x rossi"``.
    :param context: Function or plot name providing location, e.g. ``"_all_corr_data"``.

    """
    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        ts = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        logger.debug(
            " | ".join(["PROFILE", ts, f"{elapsed:.4f}", op, module, dataset, context])
        )


__all__ = ["profile_span"]
