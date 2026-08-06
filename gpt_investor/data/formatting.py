"""Markdown formatting for the higher-signal data block.

Kept separate from the fetch/scoring logic in `signals.py` so presentation
changes don't touch the data code.
"""

from __future__ import annotations


def _fmt_pct(v: float | None) -> str:
    """Format a fraction as a signed percentage, or "n/a".

    Parameters
    ----------
    v : float | None
        Fraction to format.

    Returns
    -------
    str
        Signed percentage (e.g. "+12.0%"), or "n/a" when None.
    """
    return f"{v:+.1%}" if v is not None else "n/a"


def _fmt_flow(v: float | None) -> str:
    """Format a dollar flow in signed millions, or "n/a".

    Parameters
    ----------
    v : float | None
        Dollar value to format.

    Returns
    -------
    str
        Signed millions (e.g. "+$3.2M"), or "n/a" when None.
    """
    if v is None:
        return "n/a"
    return f"{'+' if v >= 0 else '-'}${abs(v) / 1e6:.1f}M"


def format_signals(sig: dict) -> str:
    """Render a signals dict as a Markdown block for the prompt and dialog.

    Parameters
    ----------
    sig : dict
        Signals dict as returned by `signals.fetch_signals`.

    Returns
    -------
    str
        Multi-line Markdown summarising short interest, earnings, insider flow,
        multi-year trend, and peer valuation.
    """
    short = sig.get("short", {})
    peers = sig.get("peers", {})
    lines = ["**Higher-signal data**", ""]

    short_pct = short.get("short_pct")
    short_line = f"- Short interest: {short_pct:.1%} of float" if short_pct is not None else "- Short interest: n/a"
    if short.get("crowded"):
        short_line += " ⚠ crowded short (>15%)"
    lines.append(short_line)

    ed = sig.get("earnings_days")
    lines.append(f"- Next earnings: {'in ' + str(ed) + 'd' if ed is not None else 'n/a'}"
                 + (f"  ⚠ {sig['earnings_banner']}" if sig.get("earnings_banner") else ""))

    lines.append(f"- Insider net flow: 30d {_fmt_flow(sig.get('insider_30d'))}, 90d {_fmt_flow(sig.get('insider_90d'))}")

    inst = sig.get("institutional", {})
    inst_pct = inst.get("inst_pct")
    if inst_pct is not None or inst.get("n"):
        net = inst.get("net_holder_change", 0)
        direction = "accumulating" if net > 0 else "distributing" if net < 0 else "flat"
        held = f"{inst_pct:.1%}" if inst_pct is not None else "n/a"
        lines.append(
            f"- Institutional ownership: {held} held; "
            f"{inst.get('adders', 0)} adding / {inst.get('reducers', 0)} reducing "
            f"({direction}, n={inst.get('n', 0)})"
        )
    lines.append(
        f"- Multi-year trend: revenue CAGR {_fmt_pct(sig.get('rev_cagr'))}, "
        f"FCF CAGR {_fmt_pct(sig.get('fcf_cagr'))}, "
        f"op-margin slope {sig.get('op_margin_slope') if sig.get('op_margin_slope') is not None else 'n/a'}"
    )

    pe_med = peers.get("pe_median")
    if pe_med is not None:
        rel = peers.get("pe_rel") or {}
        tag = ""
        if rel.get("ratio") is not None:
            tag = f" — this name at {rel['ratio']}x the median ({'cheaper' if rel.get('cheaper') else 'richer'})"
        lines.append(f"- Peer valuation: industry median fwd P/E {pe_med} (n={peers.get('n')}){tag}")

    return "\n".join(lines)
