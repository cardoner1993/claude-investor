import reflex as rx
from loguru import logger

from gpt_investor.state import State
from gpt_investor.data.discovery import get_yf_industry_groups

_YF_INDUSTRY_GROUPS_FALLBACK: list[tuple[str, list[tuple[str, str]]]] = [
    ("Technology", [
        ("Semiconductors", "semiconductors"),
        ("Software", "software-application"),
        ("Cloud / SaaS", "software-infrastructure"),
        ("Consumer Electronics", "consumer-electronics"),
        ("Hardware", "computer-hardware"),
        ("Comms Equipment", "communication-equipment"),
    ]),
    ("Energy", [
        ("Oil & Gas", "oil-gas-integrated"),
        ("Oil Exploration", "oil-gas-e-p"),
        ("Pipelines", "oil-gas-midstream"),
    ]),
    ("Utilities", [
        ("Electric Utilities", "utilities-regulated-electric"),
        ("Independent Power", "utilities-independent-power-producers"),
        ("Gas Utilities", "utilities-regulated-gas"),
        ("Renewables", "utilities-renewable"),
        ("Solar", "solar"),
    ]),
    ("Healthcare", [
        ("Pharma", "drug-manufacturers-general"),
        ("Biotech", "biotechnology"),
        ("Medical Devices", "medical-devices"),
        ("Health Plans", "healthcare-plans"),
    ]),
    ("Financials", [
        ("Banks", "banks-diversified"),
        ("Regional Banks", "banks-regional"),
        ("Asset Management", "asset-management"),
        ("Fintech", "credit-services"),
        ("Insurance", "insurance-diversified"),
    ]),
    ("Consumer", [
        ("E-Commerce", "internet-retail"),
        ("Auto", "auto-manufacturers"),
        ("Restaurants", "restaurants"),
        ("Entertainment", "entertainment"),
    ]),
    ("Industrials", [
        ("Aerospace & Defense", "aerospace-defense"),
        ("Steel", "steel"),
        ("Chemicals", "specialty-chemicals"),
    ]),
    ("Commodities", [
        ("Gold", "gold"),
        ("Silver", "silver"),
        ("Copper", "copper"),
    ]),
    ("Communications", [
        ("Telecom", "telecom-services"),
        ("Internet & Media", "internet-content-information"),
    ]),
    ("Real Estate", [
        ("REITs", "reit-diversified"),
    ]),
]

logger.info("startup fetching YF industry taxonomy...")
_YF_INDUSTRY_GROUPS = get_yf_industry_groups() or _YF_INDUSTRY_GROUPS_FALLBACK
logger.info("startup industry groups ready: {} sectors", len(_YF_INDUSTRY_GROUPS))


@rx.memo
def analysis_html_renderer(html: rx.Var[str]) -> rx.Component:
    """Render pre-rendered analysis HTML via dangerouslySetInnerHTML.

    Separate memoized component so it gets its own JS file with useContext.

    Parameters
    ----------
    html : str
        Sanitized HTML markup to inject.

    Returns
    -------
    rx.Component
    """
    return rx.el.div(
        special_props=[rx.Var(_js_expr="{dangerouslySetInnerHTML: {__html: htmlRxMemo}}")],
        class_name="md-analysis",
    )


@rx.memo
def liquidity_html_renderer(html: rx.Var[str]) -> rx.Component:
    """Render pre-rendered liquidity HTML via dangerouslySetInnerHTML.

    Parameters
    ----------
    html : str
        Sanitized HTML markup to inject.

    Returns
    -------
    rx.Component
    """
    return rx.el.div(
        special_props=[rx.Var(_js_expr="{dangerouslySetInnerHTML: {__html: htmlRxMemo}}")],
        class_name="md-analysis",
    )


def token_counter() -> rx.Component:
    """Build the fixed bottom-right token usage chip linking to Claude usage.

    Returns
    -------
    rx.Component
    """
    return rx.el.a(
        rx.hstack(
            rx.icon("cpu", size=12, color="gray"),
            rx.text(
                State.input_tokens.to_string(), " in · ",
                State.output_tokens.to_string(), " out · ",
                State.cache_read_tokens.to_string(), " cached",
                size="1",
                color="gray",
            ),
            spacing="2",
            align="center",
        ),
        href="https://claude.ai/settings/usage",
        target="_blank",
        rel="noopener noreferrer",
        position="fixed",
        bottom="1.25em",
        right="1.25em",
        padding="0.4em 0.8em",
        background="rgba(0,0,0,0.6)",
        border="1px solid rgba(255,255,255,0.08)",
        border_radius="full",
        backdrop_filter="blur(8px)",
        text_decoration="none",
        _hover={"border": "1px solid rgba(255,255,255,0.2)"},
    )


def ticker_card(ticker_kv: list[str]) -> rx.Component:
    """Build one ticker card: name, fundamental/sentiment/Wyckoff badges, status.

    Parameters
    ----------
    ticker_kv : list[str]
        `[ticker, status]` pair from `State.tickers`.

    Returns
    -------
    rx.Component
    """
    ticker = ticker_kv[0]
    status = ticker_kv[1]
    is_done = (status == "finished") | (status == "cached")
    fund_summary = State.fund_summary[ticker]
    fund_color = State.fund_color[ticker]
    sent_summary = State.sent_summary[ticker]
    sent_color = State.sent_color[ticker]
    wyck_summary = State.wyck_summary[ticker]
    wyck_color = State.wyck_color[ticker]
    earnings = State.earnings_banner[ticker]
    audit_label = State.audit_label[ticker]
    audit_color = State.audit_color[ticker]

    return rx.card(
        rx.vstack(
            rx.heading(ticker, size="5", weight="bold"),
            rx.cond(
                earnings != "",
                rx.badge(earnings, color_scheme="red", variant="solid", radius="full", size="1"),
                rx.fragment(),
            ),
            rx.text(
                State.names[ticker_kv[0]],
                size="1",
                color="gray",
                text_align="center",
                max_width="110px",
                overflow="hidden",
                white_space="nowrap",
                text_overflow="ellipsis",
            ),
            rx.cond(
                fund_summary != "",
                rx.badge(fund_summary, color_scheme=fund_color, variant="solid", radius="full", size="1"),
                rx.fragment(),
            ),
            rx.cond(
                sent_summary != "",
                rx.badge(sent_summary, color_scheme=sent_color, variant="soft", radius="full", size="1"),
                rx.fragment(),
            ),
            rx.cond(
                wyck_summary != "",
                rx.badge(wyck_summary, color_scheme=wyck_color, variant="outline", radius="full", size="1"),
                rx.fragment(),
            ),
            rx.cond(
                audit_label != "",
                rx.badge("audit: ", audit_label, color_scheme=audit_color, variant="soft", radius="full", size="1"),
                rx.fragment(),
            ),
            rx.cond(
                status == "cached",
                rx.badge("Cached", color_scheme="blue", variant="soft", radius="full"),
                rx.cond(
                    status == "finished",
                    rx.badge("Done", color_scheme="green", variant="soft", radius="full"),
                    rx.cond(
                        status == "processing",
                        rx.hstack(rx.spinner(size="1"), rx.text("Analysing", size="1"), spacing="1", align="center"),
                        rx.cond(
                            status == "error",
                            rx.badge("Error", color_scheme="red", variant="soft", radius="full"),
                            rx.cond(
                                status == "cancelled",
                                rx.badge("Cancelled", color_scheme="gray", variant="soft", radius="full"),
                                rx.badge("Pending", color_scheme="gray", variant="soft", radius="full"),
                            ),
                        ),
                    ),
                ),
            ),
            rx.cond(
                is_done,
                rx.button(
                    "View",
                    size="1",
                    variant="ghost",
                    color_scheme="amber",
                    on_click=State.open_ticker(ticker),
                ),
                rx.box(height="22px"),
            ),
            spacing="2",
            align="center",
        ),
        width="150px",
        min_height="180px",
        display="flex",
        align_items="center",
        justify_content="center",
        cursor=rx.cond(is_done, "pointer", "default"),
        on_click=rx.cond(is_done, State.open_ticker(ticker), rx.noop()),
    )


def explainer_controls() -> rx.Component:
    """Bottom-of-dialog controls: external deep-dive links + an on-demand
    "generate plain-English summary" button (no longer auto-runs on open).

    Returns
    -------
    rx.Component
    """
    yahoo = "https://finance.yahoo.com/quote/" + State.selected_ticker
    tradingview = "https://www.tradingview.com/symbols/" + State.selected_ticker + "/"
    return rx.vstack(
        rx.divider(margin_top="0.8em", margin_bottom="0.6em"),
        rx.hstack(
            rx.link(
                rx.button(
                    rx.icon("external-link", size=13), "Yahoo Finance",
                    size="1", variant="soft", color_scheme="gray",
                ),
                href=yahoo, target="_blank",
            ),
            rx.link(
                rx.button(
                    rx.icon("line-chart", size=13), "TradingView",
                    size="1", variant="soft", color_scheme="gray",
                ),
                href=tradingview, target="_blank",
            ),
            spacing="2",
            wrap="wrap",
        ),
        rx.cond(
            State.selected_explainer_html == "",
            rx.cond(
                State.explainer_loading,
                rx.hstack(
                    rx.spinner(size="1"),
                    rx.text("Writing plain-English summary...", size="1", color="gray"),
                    spacing="2", align="center",
                ),
                rx.vstack(
                    rx.button(
                        rx.icon("message-square-text", size=14),
                        "Generate plain-English summary",
                        on_click=State.generate_explainer,
                        variant="soft", color_scheme="amber", size="1",
                    ),
                    rx.cond(
                        State.selected_explainer_failed,
                        rx.text("Couldn't generate a summary — try again.", size="1", color="gray"),
                        rx.fragment(),
                    ),
                    spacing="1", align="start",
                ),
            ),
            # Generated → render the summary right here, at the bottom, next to
            # the button that produced it (no jump to the top of the dialog).
            rx.box(
                rx.hstack(
                    rx.icon("message-square-text", size=14, color="gray"),
                    rx.text("Plain English", size="1", weight="medium", color="gray"),
                    spacing="2", align="center", padding_bottom="0.4em",
                ),
                rx.cond(
                    State.selected_explainer_partial,
                    rx.callout(
                        "AI summary — some analysis layers were unavailable, so this may be incomplete or wrong.",
                        icon="triangle-alert",
                        color_scheme="amber",
                        size="1",
                        margin_bottom="0.6em",
                    ),
                    rx.fragment(),
                ),
                analysis_html_renderer(html=State.selected_explainer_html),
                width="100%",
            ),
        ),
        spacing="2",
        align="start",
        width="100%",
        padding_top="0.5em",
    )


def analysis_dialog() -> rx.Component:
    """Build the modal that shows the selected ticker's full analysis.

    Returns
    -------
    rx.Component
    """
    return rx.dialog.root(
        rx.dialog.content(
            rx.dialog.title(
                rx.hstack(
                    rx.vstack(
                        rx.heading(State.selected_ticker, size="6", weight="bold"),
                        rx.hstack(
                            rx.text(State.selected_name, size="2", color="gray"),
                            rx.cond(
                                State.selected_is_cached,
                                rx.badge("Cached", color_scheme="blue", variant="soft", radius="full", size="1"),
                                rx.fragment(),
                            ),
                            rx.cond(
                                State.selected_earnings_banner != "",
                                rx.badge(State.selected_earnings_banner, color_scheme="red", variant="solid", radius="full", size="1"),
                                rx.fragment(),
                            ),
                            spacing="2",
                            align="center",
                        ),
                        spacing="0",
                        align="start",
                    ),
                    rx.dialog.close(
                        rx.icon_button(
                            rx.icon("x", size=16),
                            variant="ghost",
                            color_scheme="gray",
                            on_click=State.close_ticker,
                        ),
                    ),
                    justify="between",
                    align="center",
                    width="100%",
                ),
            ),
            rx.divider(),
            rx.scroll_area(
                rx.vstack(
                    rx.cond(
                        State.selected_fund_html != "",
                        rx.box(
                            rx.hstack(
                                rx.badge(
                                    State.selected_fund_summary,
                                    color_scheme=State.selected_fund_color,
                                    variant="solid",
                                    radius="full",
                                ),
                                rx.cond(
                                    State.selected_sent_summary != "",
                                    rx.badge(
                                        State.selected_sent_summary,
                                        color_scheme=State.selected_sent_color,
                                        variant="soft",
                                        radius="full",
                                    ),
                                    rx.fragment(),
                                ),
                                rx.cond(
                                    State.selected_wyck_summary != "",
                                    rx.badge(
                                        State.selected_wyck_summary,
                                        color_scheme=State.selected_wyck_color,
                                        variant="outline",
                                        radius="full",
                                    ),
                                    rx.fragment(),
                                ),
                                rx.cond(
                                    State.selected_setup_why != "",
                                    rx.badge(
                                        State.selected_setup_why,
                                        color_scheme="violet",
                                        variant="soft",
                                        radius="full",
                                    ),
                                    rx.fragment(),
                                ),
                                spacing="2",
                                padding_bottom="0.5em",
                            ),
                            analysis_html_renderer(html=State.selected_fund_html),
                            rx.cond(
                                State.selected_sent_html != "",
                                rx.box(
                                    analysis_html_renderer(html=State.selected_sent_html),
                                    padding_top="0.8em",
                                ),
                                rx.fragment(),
                            ),
                            rx.cond(
                                State.selected_wyck_html != "",
                                rx.box(
                                    analysis_html_renderer(html=State.selected_wyck_html),
                                    padding_top="0.8em",
                                ),
                                rx.fragment(),
                            ),
                            rx.cond(
                                State.selected_audit_html != "",
                                rx.box(
                                    analysis_html_renderer(html=State.selected_audit_html),
                                    padding_top="0.8em",
                                ),
                                rx.fragment(),
                            ),
                            rx.cond(
                                State.selected_signals_html != "",
                                rx.box(
                                    analysis_html_renderer(html=State.selected_signals_html),
                                    padding_top="0.8em",
                                ),
                                rx.fragment(),
                            ),
                            padding_bottom="1em",
                            border_bottom="1px solid var(--gray-a5)",
                            margin_bottom="1em",
                        ),
                        rx.fragment(),
                    ),
                    rx.cond(
                        State.selected_analysis_html != "",
                        analysis_html_renderer(html=State.selected_analysis_html),
                        rx.hstack(
                            rx.spinner(size="2"),
                            rx.text("Loading analysis...", size="2", color="gray"),
                            spacing="2", align="center", padding_top="1em",
                        ),
                    ),
                    explainer_controls(),
                    spacing="2",
                    align="start",
                ),
                height="60vh",
                padding_top="1em",
            ),
            max_width="680px",
            width="90vw",
        ),
        open=State.selected_ticker != "",
        on_open_change=State.close_ticker,
    )


def liquidity_panel() -> rx.Component:
    """Build the global liquidity card, or a spinner while it loads.

    Returns
    -------
    rx.Component
    """
    return rx.cond(
        State.liquidity_html != "",
        rx.card(
            rx.vstack(
                rx.hstack(
                    rx.icon("globe", size=14, color="gray"),
                    rx.text("Global Liquidity", size="2", weight="medium", color="gray"),
                    spacing="2",
                    align="center",
                ),
                liquidity_html_renderer(html=State.liquidity_html),
                spacing="2",
            ),
            width="100%",
            max_width="600px",
        ),
        rx.hstack(
            rx.spinner(size="1"),
            rx.text("Fetching global liquidity...", size="1", color="gray"),
            spacing="2",
            align="center",
        ),
    )


def tickers_grid() -> rx.Component:
    """Build the wrapping grid of ticker cards, or empty/loading fallbacks.

    Returns
    -------
    rx.Component
    """
    return rx.cond(
        State.tickers,
        rx.flex(
            rx.foreach(State.tickers, ticker_card),
            wrap="wrap",
            gap="3",
            justify="center",
        ),
        rx.cond(
            State.stage == "done",
            rx.text("No companies found — try a different name or ticker.", size="2", color="gray"),
            rx.hstack(rx.spinner(size="3"), rx.text("Generating tickers...", size="3", color="gray"), spacing="3"),
        ),
    )


def status_line() -> rx.Component:
    """Build the run status line plus the Stop button while analyzing.

    Returns
    -------
    rx.Component
    """
    return rx.hstack(
        rx.cond(
            State.all_done,
            rx.hstack(
                rx.icon("circle-check", size=16, color="green"),
                rx.text("All done — click any ticker to read its analysis", size="2", color="gray"),
                spacing="2",
                align="center",
            ),
            rx.cond(
                State.discovery_mode == "trending",
                rx.text("Analysing today's top setups", size="2", color="gray"),
                rx.cond(
                    State.discovery_mode == "single",
                    rx.text("Analysing ", State.company_query, size="2", color="gray"),
                    rx.text("Analysing ", State.industry, " industry", size="2", color="gray"),
                ),
            ),
        ),
        rx.cond(
            State.stage == "analyzing",
            rx.button(
                rx.icon("circle-stop", size=14),
                "Stop",
                on_click=State.cancel_analysis,
                color_scheme="red",
                variant="soft",
                size="1",
            ),
        ),
        spacing="3",
        align="center",
        justify="center",
        width="100%",
    )


def _candidate_row(cand: list[str]) -> rx.Component:
    """Build one clickable candidate row for the confirm panel.

    Parameters
    ----------
    cand : list[str]
        `[symbol, name, exchange]` triple.

    Returns
    -------
    rx.Component
    """
    ticker = cand[0]
    name = cand[1]
    exchange = cand[2]
    return rx.button(
        rx.hstack(
            rx.badge(ticker, color_scheme="amber", size="2", radius="medium"),
            rx.text(name, size="2"),
            rx.cond(
                exchange != "",
                rx.text("· ", exchange, size="1", color="gray"),
            ),
            spacing="2",
            align="center",
            width="100%",
        ),
        on_click=State.confirm_pending_with(ticker),
        variant="soft",
        color_scheme="gray",
        size="2",
        width="100%",
        justify="start",
    )


def confirm_resolution_panel() -> rx.Component:
    """Build the "Did you mean…?" candidate picker for ambiguous queries.

    Shown when the user typed a company name and we resolved candidates but
    haven't started the analysis yet. They pick one or edit the query.

    Returns
    -------
    rx.Component
    """
    return rx.cond(
        State.stage == "confirming",
        rx.vstack(
            rx.text("Did you mean…?", size="2", color="gray", weight="medium"),
            rx.foreach(State.pending_candidates, _candidate_row),
            rx.button(
                "Edit query",
                on_click=State.cancel_pending,
                color_scheme="gray",
                variant="ghost",
                size="1",
            ),
            spacing="2",
            align="stretch",
            padding="3",
            border="1px solid var(--gray-a5)",
            border_radius="6px",
            width="100%",
            max_width="600px",
        ),
    )


def hero() -> rx.Component:
    """Build the page header: title, tagline, disclaimer badge.

    Returns
    -------
    rx.Component
    """
    return rx.vstack(
        rx.heading("Claude Investor", size="8", weight="bold"),
        rx.text("AI-powered investment analysis · powered by Claude Code", size="2", color="gray"),
        rx.badge("Not financial advice", color_scheme="amber", variant="surface", radius="full"),
        spacing="2",
        align="center",
    )


def industry_groups() -> rx.Component:
    """Build the collapsible sector accordion of clickable industry chips.

    Returns
    -------
    rx.Component
    """
    return rx.vstack(
        *[
            rx.vstack(
                rx.hstack(
                    rx.cond(
                        State.expanded_sectors.contains(sector),
                        rx.icon("chevron-down", size=12, color="gray"),
                        rx.icon("chevron-right", size=12, color="gray"),
                    ),
                    rx.text(sector, size="1", weight="medium", color="gray"),
                    rx.text(
                        f"({len(industries)})",
                        size="1",
                        color="gray",
                        opacity="0.4",
                    ),
                    spacing="1",
                    align="center",
                    cursor="pointer",
                    on_click=State.toggle_sector(sector),
                    _hover={"opacity": "0.7"},
                ),
                rx.cond(
                    State.expanded_sectors.contains(sector),
                    rx.flex(
                        *[
                            rx.badge(
                                label,
                                color_scheme="gray",
                                variant="surface",
                                radius="full",
                                cursor=rx.cond(State.stage == "analyzing", "not-allowed", "pointer"),
                                opacity=rx.cond(State.stage == "analyzing", "0.4", "1"),
                                on_click=rx.cond(
                                    State.stage == "analyzing",
                                    rx.noop(),
                                    State.industry_pick(label, yf_key),
                                ),
                                _hover={"opacity": rx.cond(State.stage == "analyzing", "0.4", "0.8")},
                            )
                            for label, yf_key in industries
                        ],
                        wrap="wrap",
                        gap="2",
                    ),
                ),
                spacing="1",
                align="start",
                width="100%",
            )
            for sector, industries in _YF_INDUSTRY_GROUPS
        ],
        spacing="2",
        width="100%",
        align="start",
    )


def search_form() -> rx.Component:
    """Build the main search card: trending buttons, taxonomy, and text inputs.

    Returns
    -------
    rx.Component
    """
    return rx.card(
        rx.vstack(
            rx.hstack(
                rx.button(
                    rx.icon("trending-up", size=14),
                    "Today's Setups",
                    on_click=State.trending_pick,
                    loading=(State.stage == "analyzing"),
                    variant="surface",
                    color_scheme="violet",
                    size="2",
                    flex="1",
                ),
                rx.button(
                    rx.icon("bar-chart-2", size=14),
                    "Trending Industries",
                    on_click=State.fetch_trending_industries,
                    loading=State.trending_industries_loading,
                    disabled=(State.stage == "analyzing"),
                    variant="surface",
                    color_scheme="orange",
                    size="2",
                    flex="1",
                ),
                spacing="2",
                width="100%",
            ),
            rx.cond(
                State.trending_industries,
                rx.flex(
                    rx.foreach(
                        State.trending_industries,
                        lambda item: rx.badge(
                            item[0],
                            color_scheme="orange",
                            variant="soft",
                            radius="full",
                            cursor=rx.cond(State.stage == "analyzing", "not-allowed", "pointer"),
                            opacity=rx.cond(State.stage == "analyzing", "0.4", "1"),
                            on_click=rx.cond(
                                State.stage == "analyzing",
                                rx.noop(),
                                State.industry_pick(item[0], item[1]),
                            ),
                            _hover={"opacity": rx.cond(State.stage == "analyzing", "0.4", "0.8")},
                        ),
                    ),
                    wrap="wrap",
                    gap="2",
                ),
            ),
            rx.divider(),
            industry_groups(),
            rx.divider(),
            rx.form(
                rx.hstack(
                    rx.input(
                        placeholder="Custom industry...",
                        id="industry",
                        value=State.industry_input,
                        on_change=State.set_industry_input,
                        width="100%",
                        size="1",
                    ),
                    rx.button(
                        "Go",
                        type="submit",
                        loading=(State.stage == "analyzing"),
                        color_scheme="gray",
                        size="1",
                    ),
                    width="100%",
                    spacing="2",
                ),
                on_submit=State.handle_submit,
            ),
            rx.form(
                rx.hstack(
                    rx.icon("search", size=13, color="gray", flex_shrink="0"),
                    rx.input(
                        placeholder="Ticker or company name (e.g. CEG, Constellation Energy)...",
                        id="company",
                        value=State.company_query,
                        on_change=State.set_company_query,
                        width="100%",
                        size="1",
                        variant="soft",
                    ),
                    rx.button(
                        "Analyse",
                        type="submit",
                        loading=(State.stage == "analyzing"),
                        color_scheme="amber",
                        size="1",
                    ),
                    width="100%",
                    spacing="2",
                    align="center",
                ),
                on_submit=State.handle_company_submit,
            ),
            rx.cond(
                State.error_message != "",
                rx.text(State.error_message, size="1", color="red"),
            ),
            spacing="3",
        ),
        width="100%",
        max_width="600px",
    )
