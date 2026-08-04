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
    """Call `fn` with exponential backoff, re-raising the last error on total failure.

    Delay grows by `backoff` after each miss. `_sleep` is injectable so tests
    don't actually wait.

    Parameters
    ----------
    fn : Callable[..., T]
        Callable to invoke; `*args`/`**kwargs` are forwarded to it.
    tries : int
        Max attempts. Optional, default 3.
    base_delay : float
        Seconds to wait after the first failure. Optional, default 0.5.
    backoff : float
        Multiplier applied to the delay after each failed attempt. Optional, default 2.0.
    exceptions : tuple[type[BaseException], ...]
        Exception types treated as retryable. Optional, default (Exception,).
    label : str or None
        Name used in log lines; falls back to `fn.__name__`. Optional, default None.
    _sleep : Callable[[float], None]
        Sleep function, injectable for tests. Optional, default time.sleep.

    Returns
    -------
    T
        The value returned by the first successful call.
    """
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

    Parameters
    ----------
    obj : Any
        Root object to traverse.
    *path
        Sequence of keys/indices to descend.
    default : Any
        Value returned on any miss. Optional, default None.

    Returns
    -------
    Any
        The nested value, or `default` if any step misses.
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

    Parameters
    ----------
    seq : Any
        A dict (returned as-is), a list/tuple to scan, or anything else.

    Returns
    -------
    dict
        The sequence itself if a dict, the first dict element, or `{}`.
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
    """Clear the per-run degradation set.

    Call at the start of a run so `degraded_legs()` reflects only this run.
    """
    with _lock:
        _degraded.clear()


def degraded_legs() -> list[str]:
    """Sorted names of legs that served last-good this run.

    Returns
    -------
    list[str]
        Leg names marked degraded since the last `reset_health()`.
    """
    with _lock:
        return sorted(_degraded)


def _mark_degraded(leg: str) -> None:
    """Record `leg` as degraded for this run.

    Parameters
    ----------
    leg : str
        Leg name to add to the degradation set.
    """
    with _lock:
        _degraded.add(leg)


def resilient(leg: str, fn: Callable[..., T], *args, key=None, **kwargs) -> T:
    """Run a critical leg with retry, serving the last-good value on total failure.

    On success the value is cached under `(leg, key)`. On total failure the leg
    is marked degraded and the cached value is served if present; otherwise the
    error re-raises. `key` defaults to the positional args so per-ticker legs
    cache per ticker.

    Parameters
    ----------
    leg : str
        Leg name for degradation tracking and last-good caching.
    fn : Callable[..., T]
        Callable to run under `with_retry`; `*args`/`**kwargs` are forwarded.
    key : Any
        Cache discriminator; falls back to `args`. Optional, default None.

    Returns
    -------
    T
        The fresh value, or the last-good value when the leg degrades.
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
