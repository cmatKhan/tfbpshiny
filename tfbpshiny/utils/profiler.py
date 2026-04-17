"""Timing context manager for the profiler logger."""

from __future__ import annotations

import json
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
    Time a block of code and emit one structured JSON record to *logger*.

    Each record has the following keys::

        {"event": "PROFILE", "ts": "...", "elapsed_s": 0.1234,
         "op": "...", "module": "...", "dataset": "...",
         "context": "...", "session_id": "..."}

    :param logger: The ``"profiler"`` logger.
    :param op: Operation label, e.g. ``"vdb.query"`` or ``"plot.build"``.
    :param module: Shiny module name, e.g. ``"binding"``. Empty for init-time spans.
    :param dataset: Dataset(s) involved, e.g. ``"harbison"`` or ``"harbisонxrossi"``.
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
            json.dumps(
                {
                    "event": "PROFILE",
                    "ts": ts,
                    "elapsed_s": round(elapsed, 4),
                    "op": op,
                    "module": module,
                    "dataset": dataset,
                    "context": context,
                    "session_id": session_id,
                }
            )
        )


def log_session_event(logger: Logger, event: str, session_id: str) -> None:
    """
    Emit a session lifecycle record to *logger*.

    Each record has the following keys::

        {"event": "SESSION", "ts": "...",
        "lifecycle": "START"|"END", "session_id": "..."}

    :param logger: The ``"profiler"`` logger.
    :param event: ``"START"`` or ``"END"``.
    :param session_id: Shiny session ID.

    """
    ts = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    logger.debug(
        json.dumps(
            {
                "event": "SESSION",
                "ts": ts,
                "lifecycle": event,
                "session_id": session_id,
            }
        )
    )


__all__ = ["profile_span", "log_session_event"]
