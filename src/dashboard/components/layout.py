import dash
from dash import Dash, html
import dash_bootstrap_components as dbc

def create_layout(app: Dash) -> html.Div:
    """
    Creates the main layout for the Dash application.

    Args:
        app (Dash): The Dash application instance.

    Returns:
        html.Div: The main layout component.
    """
    return html.Div(
        [
            dbc.NavbarSimple(
                [
                    dbc.NavLink(f"{page['name']}", href=page["relative_path"])
                    for page in dash.page_registry.values()
                ],
                brand="desercionAR",
                brand_href="/",
                color="primary",
                dark=True,
                sticky="top",
            ),
            dash.page_container,
        ]
    )
