"""Loguru setup. Called once at app start from gpt_investor.py.

Console: colored, INFO+, human-readable.
File:    logs/claude-investor.log, DEBUG+, rotated at 2 MB, kept 3 backups.
         Contains full LLM prompts/responses for audit.

Override level via env: GPT_INVESTOR_LOG_LEVEL=DEBUG reflex run
"""

import os
import sys
from pathlib import Path

from loguru import logger

_initialized = False


def _render_extra(extra: dict) -> str:
    """Render bind() context as `key=value key=value`.

    No braces — loguru would re-interpret them as format placeholders and
    raise KeyError.

    Parameters
    ----------
    extra : dict
        The record's bound context.

    Returns
    -------
    str
        Space-prefixed key=value pairs, or empty string when no context.
    """
    if not extra:
        return ""
    return " " + " ".join(f"{k}={v}" for k, v in extra.items())


def _console_format(record) -> str:
    """Build the colored console format string for a log record.

    Parameters
    ----------
    record : loguru.Record
        The record being formatted.

    Returns
    -------
    str
        Loguru format template with the record's extra context inlined.
    """
    extra_part = _render_extra(record["extra"])
    if extra_part:
        extra_part = f" <yellow>{extra_part.strip()}</yellow>"
    return (
        "<green>{time:HH:mm:ss}</green> "
        "<level>{level: <7}</level> "
        "<cyan>{name}:{function}</cyan>"
        + extra_part
        + " <level>{message}</level>\n"
    )


def _file_format(record) -> str:
    """Build the plain file format string for a log record.

    Parameters
    ----------
    record : loguru.Record
        The record being formatted.

    Returns
    -------
    str
        Loguru format template with the record's extra context inlined.
    """
    return (
        "{time:HH:mm:ss.SSS} {level: <7} [{name}:{function}:{line}]"
        + _render_extra(record["extra"])
        + " {message}\n"
    )


def setup_logging(
    level: str | None = None,
    log_file: str | os.PathLike = "logs/claude-investor.log",
) -> None:
    """Configure loguru sinks once per process (console + rotating file).

    Idempotent — repeat calls after the first are no-ops.

    Parameters
    ----------
    level : str, optional
        Console log level; defaults to `GPT_INVESTOR_LOG_LEVEL` env or INFO.
    log_file : str | os.PathLike, optional
        Path to the DEBUG file sink; default "logs/claude-investor.log".
    """
    global _initialized
    if _initialized:
        return

    level = (level or os.environ.get("GPT_INVESTOR_LOG_LEVEL", "INFO")).upper()

    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format=_console_format,
        colorize=True,
        backtrace=True,
        diagnose=False,
    )

    try:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_file,
            level="DEBUG",
            rotation="2 MB",
            retention=3,
            encoding="utf-8",
            format=_file_format,
            backtrace=True,
            diagnose=True,
            # No enqueue=True — only useful for cross-process logging.
            # Threads are already serialized by loguru's internal lock, and
            # multiprocessing.Queue leaks semaphores on abrupt shutdown
            # (reflex hot reload, Ctrl+C).
        )
    except OSError as e:
        logger.warning("could not open log file {}: {} — console only", log_file, e)

    _initialized = True
    logger.info("logging initialized  console={}  file={}", level, log_file)
