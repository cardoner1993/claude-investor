# Claude Investor

Based on the [gpt-investor](https://github.com/mshumer/gpt-investor) by
[Matt](https://twitter.com/mattshumer_). See the [original README](notebooks/README.md) for more details.

This is an app built using [Reflex](https://github.com/reflex-dev/reflex). The AI prompts, data fetching and processing logic are directly lifted from the [notebook](notebooks/Claude_Investor.ipynb). The UI elements and the reactivity are built using Reflex.

## Prerequisites

You need [Claude Code CLI](https://claude.ai/code) installed and logged in. No API key required — the app calls the `claude` CLI directly, which uses your existing Claude Code session for authentication.

```bash
# Install Claude Code if you haven't already, then log in:
claude login
```

## Requirements

Managed via Poetry (`pyproject.toml`). Install with:

```bash
poetry install
```

## Environment variables

The app reads one optional environment variable `MAX_TICKERS_TO_ANALYZE`. It controls how many tickers Claude searches for per industry, defaults to `4`.

## Running the app

`reflex` is installed inside the `claude-investor` pyenv virtualenv, so activate it first:

```bash
pyenv activate claude-investor
```

Then initialise (first time only) and run:

```bash
reflex init
reflex run
```

To pass environment variables:

```bash
MAX_TICKERS_TO_ANALYZE=6 reflex run
```

Also check out [Reflex Documentation](https://reflex.dev/docs/getting-started/introduction/) to build/run/host your own app.

## Background | Excerpt from the original README

Below is part of the original README on the implementation.

### Claude-Investor | The first release in the gpt-investor repo

Claude-Investor is an experimental investment analysis agent that utilizes the Claude 3 Opus and Haiku models to provide comprehensive insights and recommendations for stocks in a given industry. It retrieves financial data, news articles, and analyst ratings for companies, performs sentiment analysis, and generates comparative analyses to rank the companies based on their investment potential.

### Workflow

- Generates a list of ticker symbols for major companies in a specified industry
- Retrieves historical price data, balance sheets, financial statements, and news articles for each company
- Performs sentiment analysis on news articles to gauge market sentiment
- Retrieves analyst ratings and price targets for each company
- Conducts industry and sector analysis to understand market trends and competitive landscape
- Generates comparative analyses between the selected company and its peers
- Provides a final investment recommendation for each company based on the comprehensive analysis, including price targets
- Ranks the companies within the industry based on their investment attractiveness

### Disclaimer

Claude-Investor is an educational and informational tool designed to assist in investment analysis. It should not be considered as financial advice or a substitute for professional investment guidance. Always conduct thorough research and consult with a qualified financial advisor before making any investment decisions.
