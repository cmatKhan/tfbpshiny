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
    session_id: str = "",
) -> Generator[None]:
    """
    Time a block of code and emit one fixed-schema pipe-delimited record to *logger*.

    The record always has exactly eight pipe-separated columns so it can be read
    without regex using ``pandas.read_csv(sep="|")``:

    ::

        PROFILE | timestamp | elapsed_s | op | module | dataset | context | session_id

    :param logger: The ``"profiler"`` logger.
    :param op: Operation label, e.g. ``"vdb.query"`` or ``"plot.build"``.
    :param module: Shiny module name, e.g. ``"binding"``. Empty for init-time spans.
    :param dataset: Dataset(s) involved, e.g. ``"harbison"`` or ``"harbison x rossi"``.
    :param context: Function or plot name providing location, e.g. ``"_all_corr_data"``.
    :param session_id: Shiny session ID, used to correlate spans and count concurrent
        users. Empty for init-time spans that run outside a session.

    """
    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        ts = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        logger.debug(
            " | ".join(
                [
                    "PROFILE",
                    ts,
                    f"{elapsed:.4f}",
                    op,
                    module,
                    dataset,
                    context,
                    session_id,
                ]
            )
        )


def log_session_event(logger: Logger, event: str, session_id: str) -> None:
    """
    Emit a SESSION_START or SESSION_END record to *logger*.

    Record format (five pipe-separated columns)::

        SESSION | timestamp | event | session_id

    :param logger: The ``"profiler"`` logger.
    :param event: ``"START"`` or ``"END"``.
    :param session_id: Shiny session ID.

    """
    ts = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    logger.debug(" | ".join(["SESSION", ts, event, session_id]))


__all__ = ["profile_span", "log_session_event"]
