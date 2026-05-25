import reflex as rx

config = rx.Config(
    app_name="gpt_investor",
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