import reflex as rx

config = rx.Config(
    app_name="gpt_investor",
    # Single-page analysis app — no sitemap needed; silences the default-plugin warning.
    disable_plugins=[rx.plugins.SitemapPlugin],
    plugins=[
        rx.plugins.RadixThemesPlugin(
            theme=rx.theme(
                appearance="dark",
                has_background=True,
                radius="large",
                accent_color="gold",
            )
        ),
    ],
)