#!/usr/bin/env python
"""Calibration report over verdict_history.

Groups filled verdicts by fund_tier / verdict / sentiment_conf / regime_label /
wyckoff_phase and prints, per group: N, mean forward return, hit rate (return
> 0), and mean alpha vs SPY over the same window. Answers "was the yes/no
right?" — the whole point of the feedback loop.

    python scripts/calibration.py --horizon 30 --prompt-version v2
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gpt_investor.storage.cache import all_verdicts
from gpt_investor.llm.verdict import PROMPT_VERSION

_GROUP_FIELDS = ["fund_tier", "verdict", "sentiment_conf", "regime_label", "wyckoff_phase"]


def _ret(a: float | None, b: float | None) -> float | None:
    """Simple return `(b - a) / a`, or None if either endpoint is missing.

    Parameters
    ----------
    a : float or None
        Start value (denominator).
    b : float or None
        End value.

    Returns
    -------
    float or None
        The fractional return, or None when `a` is missing/zero or `b` is missing.
    """
    if a is None or b is None or a == 0:
        return None
    return (b - a) / a


def _rows_with_return(verdicts: list[dict], horizon: int) -> list[dict]:
    """Keep verdicts whose horizon outcome is filled, annotating return + alpha.

    Rows without a computable forward return are dropped; `_alpha` is None when
    the SPY benchmark is unavailable.

    Parameters
    ----------
    verdicts : list of dict
        Raw verdict-history rows.
    horizon : int
        Forward window in days (selects the `price_Nd` / `spy_Nd` columns).

    Returns
    -------
    list of dict
        The surviving rows, each with added `_ret` and `_alpha` keys.
    """
    price_col, spy_col = f"price_{horizon}d", f"spy_{horizon}d"
    out = []
    for v in verdicts:
        r = _ret(v.get("price"), v.get(price_col))
        if r is None:
            continue
        spy_r = _ret(v.get("spy_at_capture"), v.get(spy_col))
        out.append({**v, "_ret": r, "_alpha": (r - spy_r) if spy_r is not None else None})
    return out


def _agg(rows: list[dict]) -> tuple[int, float, float, float | None]:
    """Aggregate a group of annotated rows into summary stats.

    Parameters
    ----------
    rows : list of dict
        Rows carrying `_ret` and `_alpha`, as produced by `_rows_with_return`.

    Returns
    -------
    tuple of (int, float, float, float or None)
        Count, mean return, hit rate (fraction with return > 0), and mean alpha
        (None when no row has a benchmark).
    """
    n = len(rows)
    mean_ret = sum(r["_ret"] for r in rows) / n
    hit = sum(1 for r in rows if r["_ret"] > 0) / n
    alphas = [r["_alpha"] for r in rows if r["_alpha"] is not None]
    mean_alpha = (sum(alphas) / len(alphas)) if alphas else None
    return n, mean_ret, hit, mean_alpha


def _print_group(field: str, rows: list[dict]) -> None:
    """Print a per-value breakdown of stats grouped by one field.

    Buckets rows by `field` (missing values shown as "—") and prints N, mean
    return, hit rate, and alpha for each, largest bucket first.

    Parameters
    ----------
    field : str
        Row key to group by.
    rows : list of dict
        Annotated rows from `_rows_with_return`.
    """
    buckets: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        buckets[r.get(field) or "—"].append(r)
    print(f"\n== by {field} ==")
    print(f"{'value':<22}{'N':>4}{'mean ret':>10}{'hit rate':>10}{'alpha':>10}")
    for value, brows in sorted(buckets.items(), key=lambda x: -len(x[1])):
        n, mean_ret, hit, alpha = _agg(brows)
        alpha_s = f"{alpha:+.2%}" if alpha is not None else "n/a"
        print(f"{value:<22}{n:>4}{mean_ret:>+10.2%}{hit:>10.0%}{alpha_s:>10}")


def main() -> None:
    """Print the calibration report for one horizon and prompt version.

    Parses CLI args (`--horizon`, `--prompt-version`), loads verdict history,
    filters to the chosen prompt contract, and prints overall plus per-group
    stats.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=30, choices=[7, 30, 90, 365])
    ap.add_argument("--prompt-version", default=PROMPT_VERSION,
                    help="filter to one prompt contract, or 'all'")
    args = ap.parse_args()

    verdicts = all_verdicts()
    if args.prompt_version != "all":
        verdicts = [v for v in verdicts if v.get("prompt_version") == args.prompt_version]

    rows = _rows_with_return(verdicts, args.horizon)
    print(f"verdict_history: {len(verdicts)} rows (prompt={args.prompt_version}), "
          f"{len(rows)} with {args.horizon}d outcome filled")
    if not rows:
        print("nothing to report yet — let the nightly filler run once the horizon elapses.")
        return

    n, mean_ret, hit, alpha = _agg(rows)
    alpha_s = f"{alpha:+.2%}" if alpha is not None else "n/a"
    print(f"\noverall: N={n}  mean ret={mean_ret:+.2%}  hit rate={hit:.0%}  alpha={alpha_s}")
    for field in _GROUP_FIELDS:
        _print_group(field, rows)


if __name__ == "__main__":
    main()
