"""yfinance resilience layer.

Every analytical layer rides one unofficial, rate-limited, schema-drifting
source. This module centralises the defences so a transient Yahoo hiccup
degrades one leg instead of taking the whole run dark:

    with_retry      — retry with exponential backoff on transient errors
    safe_get        — tolerant nested dict/list access (missing keys → default)
    first_dict      — pull the first dict out of a heterogeneous list
    resilient       — retry + serve-last-good + degradation tracking, for the
                      critical legs (price, fundamentals)
    health snapshot — which legs degraded this run (surfaced in the run log)

Pure helpers (`safe_get`, `first_dict`) unit-test without any network.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, TypeVar

from loguru import logger

T = TypeVar("T")

_MISS = object()


def with_retry(
    fn: Callable[..., T],
    *args,
    tries: int = 3,
    base_delay: float = 0.5,
    backoff: float = 2.0,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
    label: str | None = None,
    _sleep: Callable[[float], None] = time.sleep,
    **kwargs,
) -> T:
    """Call `fn` with exponential backoff. Re-raises the last error if every
    attempt fails. `_sleep` is injectable so tests don't actually wait."""
    name = label or getattr(fn, "__name__", "call")
    delay = base_delay
    last: BaseException | None = None
    for attempt in range(1, tries + 1):
        try:
            return fn(*args, **kwargs)
        except exceptions as e:
            last = e
            if attempt == tries:
                break
            logger.debug("{} attempt {}/{} failed: {}; retry in {:.1f}s", name, attempt, tries, e, delay)
            _sleep(delay)
            delay *= backoff
    logger.warning("{} failed after {} tries: {}", name, tries, last)
    raise last  # type: ignore[misc]


def safe_get(obj: Any, *path, default=None):
    """Walk `path` through nested dicts/lists, returning `default` on any miss.

    Centralises the scattered `isinstance(x, dict)` / index-bounds checks. Int
    keys index into lists/tuples; anything else keys into dicts.
    """
    cur = obj
    for key in path:
        if isinstance(cur, dict):
            cur = cur.get(key, _MISS)
        elif isinstance(cur, (list, tuple)) and isinstance(key, int) and -len(cur) <= key < len(cur):
            cur = cur[key]
        else:
            return default
        if cur is _MISS:
            return default
    return cur


def first_dict(seq: Any) -> dict:
    """First dict element of a heterogeneous sequence, or `{}`.

    Guards the known `tool_use_result.results` gotcha where a trailing element
    is a bare prose string rather than a dict.
    """
    if isinstance(seq, dict):
        return seq
    if isinstance(seq, (list, tuple)):
        for el in seq:
            if isinstance(el, dict):
                return el
    return {}


# --- degradation tracking + serve-last-good --------------------------------

_lock = threading.Lock()
_last_good: dict = {}
_degraded: set[str] = set()


def reset_health() -> None:
    """Clear the per-run degradation set (call at the start of a run)."""
    with _lock:
        _degraded.clear()


def degraded_legs() -> list[str]:
    with _lock:
        return sorted(_degraded)


def _mark_degraded(leg: str) -> None:
    with _lock:
        _degraded.add(leg)


def resilient(leg: str, fn: Callable[..., T], *args, key=None, **kwargs) -> T:
    """Run a critical leg with retry; on total failure, serve the last good
    value for `(leg, key)` if we have one, marking the leg degraded. Re-raises
    only when there is no cached fallback.

    `key` defaults to the positional args (so per-ticker legs cache per ticker).
    """
    cache_key = (leg, key if key is not None else args)
    try:
        val = with_retry(fn, *args, label=leg, **kwargs)
    except Exception:
        _mark_degraded(leg)
        with _lock:
            cached = _last_good.get(cache_key, _MISS)
        if cached is not _MISS:
            logger.warning("{} degraded — serving last-good value", leg)
            return cached
        raise
    with _lock:
        _last_good[cache_key] = val
    return val
