"""Welcome to Reflex! This file outlines the steps to create a basic app."""

from dotenv import load_dotenv

load_dotenv()  # populate os.environ from .env before anything reads it

import reflex as rx

from gpt_investor.infra.logging_config import setup_logging

setup_logging()

from gpt_investor.ui.components import (
    analysis_dialog,
    hero,
    liquidity_panel,
    search_form,
    status_line,
    tickers_grid,
    token_counter,
)
from gpt_investor.state import State


def index() -> rx.Component:
    return rx.box(
        token_counter(),
        analysis_dialog(),
        rx.vstack(
            hero(),
            search_form(),
            rx.button(
                "Test UI",
                on_click=State.load_mock_data,
                size="1",
                variant="ghost",
                color_scheme="gray",
            ),
            rx.cond(
                (State.stage == "analyzing") | (State.stage == "done"),
                rx.vstack(
                    status_line(),
                    liquidity_panel(),
                    tickers_grid(),
                    spacing="4",
                    align="center",
                    width="100%",
                ),
            ),
            align="center",
            spacing="8",
            width="min(900px, 92vw)",
            padding_y="5em",
        ),
        display="flex",
        justify_content="center",
        min_height="100vh",
    )


app = rx.App(stylesheets=["/analysis.css"])
app.add_page(index, title="Claude Investor")
