"""Low-level coercion + yfinance-statement extraction helpers.

Shared by the signal fetchers so the DataFrame plumbing lives in one place,
separate from the scoring logic.
"""

from __future__ import annotations

import math
from datetime import date


def _safe_float(x) -> float | None:
    """Coerce a value to a finite float, or None.

    None when the value can't be parsed or is NaN/inf.

    Parameters
    ----------
    x : object
        Value to coerce.

    Returns
    -------
    float | None
        Finite float, or None on failure.
    """
    try:
        v = float(x)
        return v if not (math.isnan(v) or math.isinf(v)) else None
    except (TypeError, ValueError):
        return None


def _as_date(x) -> date | None:
    """Coerce a value or pandas Timestamp to a `date`, or None.

    Parameters
    ----------
    x : object
        Value to coerce; `date` passes through, Timestamp is downcast.

    Returns
    -------
    date | None
        A `date`, or None when the value can't be converted.
    """
    if isinstance(x, date):
        return x
    try:
        return x.date()  # pandas Timestamp
    except AttributeError:
        return None


def _row(df, label: str) -> list[float] | None:
    """Clean row of a yfinance statement DataFrame, newest→oldest.

    Use for standalone series (CAGR endpoints) where position alignment across
    rows doesn't matter.

    Parameters
    ----------
    df : pandas.DataFrame
        Statement frame indexed by line-item label.
    label : str
        Row label to extract.

    Returns
    -------
    list[float] | None
        Floats with NaN dropped, or None when the row is absent or empty.
    """
    raw = _raw_row(df, label)
    if raw is None:
        return None
    return [v for v in raw if v is not None] or None


def _raw_row(df, label: str) -> list | None:
    """Raw row, newest→oldest, with NaN kept as None (positions preserved).

    Use when two rows must stay column-aligned before dropping missing cells.

    Parameters
    ----------
    df : pandas.DataFrame
        Statement frame indexed by line-item label.
    label : str
        Row label to extract.

    Returns
    -------
    list | None
        Floats/None per column, or None when the row is absent or extraction
        fails.
    """
    try:
        if label not in df.index:
            return None
        return [_safe_float(v) for v in df.loc[label].tolist()]
    except Exception:
        return None
