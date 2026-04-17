import logging
from enum import Enum
from typing import Literal

_PROFILE_FORMAT = "%(message)s"


class LogLevel(Enum):
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL

    @classmethod
    def from_string(cls, level_str: str) -> Enum:
        """
        Convert a string representation of a log level to a LogLevel enum.

        :param level_str: The string representation of the log level.
        :return: The corresponding LogLevel enum.
        :raises ValueError: If the log level string is not recognized.

        """
        try:
            return getattr(cls, level_str.upper())
        except AttributeError:
            raise ValueError(
                f"Invalid log level: {level_str}. "
                f"Choose from {', '.join(cls._member_names_)}."
            )


def configure_logger(
    name: str,
    level: int = logging.DEBUG,
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handler_type: Literal["console", "file"] = "console",
    log_file: str = "tfbpmodeling.log",
) -> logging.Logger:
    """
    Configure a logger with a single handler.

    :param name: Logger name.
    :param level: Logging level (``logging.DEBUG``, ``logging.INFO``, etc.).
    :param format: Log record format string.
    :param handler_type: Destination — ``"console"`` or ``"file"``.
    :param log_file: Path used when ``handler_type="file"``.
    :returns: Configured logger.
    :rtype: logging.Logger
    :raises ValueError: If any parameter is invalid.

    """
    if not isinstance(name, str):
        raise ValueError("name must be a string")
    if not isinstance(level, int):
        raise ValueError("level must be an integer")
    if level not in [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR]:
        raise ValueError("Invalid logging level")
    if not isinstance(format, str):
        raise ValueError("format must be a string")
    if handler_type not in ("console", "file"):
        raise ValueError("handler_type must be 'console' or 'file'")
    if handler_type == "file" and not log_file:
        raise ValueError("log_file must be specified for file handler")

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers to avoid duplicates on re-configuration.
    for h in logger.handlers[:]:
        logger.removeHandler(h)

    if handler_type == "console":
        handler: logging.Handler = logging.StreamHandler()
    else:
        handler = logging.FileHandler(log_file)

    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(format))
    logger.addHandler(handler)

    return logger


def configure_profile_logger(
    handler_type: Literal["console", "file"] = "console",
    level: int = logging.DEBUG,
    log_file: str = "tfbpshiny_profile.log",
    enabled: bool = True,
) -> logging.Logger:
    """
    Configure and return the ``"profiler"`` logger for timing instrumentation.

    The profiler logger uses a bare ``%(message)s`` format because all
    structure is embedded in the message by
    :func:`tfbpshiny.utils.profiler.profile_span`.
    It never propagates to the root or ``"shiny"`` logger.

    :param handler_type: Destination for profile records — ``"console"`` or ``"file"``.
    :param level: Log level; ignored (set to ``CRITICAL``) when ``enabled=False``.
    :param log_file: Path used when ``handler_type="file"``.
    :param enabled: When ``False``, silences the logger by setting its level
        to ``CRITICAL``.
    :returns: Configured ``"profiler"`` logger.
    :rtype: logging.Logger

    """
    effective_level = level if enabled else logging.CRITICAL
    logger = configure_logger(
        "profiler",
        level=effective_level,
        format=_PROFILE_FORMAT,
        handler_type=handler_type,
        log_file=log_file,
    )
    logger.propagate = False
    return logger
