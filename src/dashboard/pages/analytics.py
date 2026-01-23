import dash
from dash import dcc, html, callback, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from src.dashboard.utils.data_loader import load_train_data
from src.dashboard.utils.translations import (
    ANALYTICS_TITLE,
    ANALYTICS_DROPDOWN_PLACEHOLDER,
    VARIABLE_DESCRIPTIONS,
)

dash.register_page(__name__)

data = load_train_data()

def get_key_from_value(dictionary, query):
    if query in dictionary.values():
        return next(key for key, value in dictionary.items() if value == query)
    return None

def make_violin_plot(df: pd.DataFrame, col: str = "CH06", bivariate: str = "DESERTO") -> go.Figure:
    data_0 = df[col][df[bivariate] == 0]
    data_1 = df[col][df[bivariate] == 1]
    fig = go.Figure()
    fig.add_trace(
        go.Violin(
            x=df[bivariate][df[bivariate] == 0],
            y=data_0,
            legendgroup="No Dropout",
            scalegroup="0",
            name="No Dropout",
            width=0.3,
            points=False,
            side="negative",
            pointpos=-1.5,
            line_color="lightseagreen",
        )
    )
    fig.add_trace(
        go.Violin(
            x=df[bivariate][df[bivariate] == 1],
            y=data_1,
            legendgroup="Dropout",
            scalegroup="1",
            name="Dropout",
            width=0.3,
            points=False,
            side="positive",
            pointpos=1.5,
            line_color="mediumpurple",
        )
    )
    min_value = df[col].min()
    fig.update_layout(
        violingap=0,
        violinmode="overlay",
        xaxis_showticklabels=False,
        yaxis={"range": [min_value - 2, df[col].max() + 1]},
    )
    return fig

def make_hist(df: pd.DataFrame, col: str = "CH06", color_col: str = "DESERTO") -> px.histogram:
    fig = px.histogram(data_frame=df, x=col, color=color_col)
    return fig

layout = dbc.Container(
    [
        dbc.Row(
            [
                html.H2(ANALYTICS_TITLE),
                dcc.Dropdown(
                    id="variable-dropdown",
                    options=[
                        {"label": desc, "value": desc}
                        for desc in VARIABLE_DESCRIPTIONS.values()
                    ],
                    placeholder=ANALYTICS_DROPDOWN_PLACEHOLDER,
                ),
            ]
        ),
        dbc.Row(
            [
                dbc.Col(dcc.Graph(id="histplot"), width=6),
                dbc.Col(dcc.Graph(id="violin-plot"), width=6),
            ]
        ),
    ],
    fluid=True,
)

@callback(
    Output("violin-plot", "figure"),
    [Input("variable-dropdown", "value")],
    prevent_initial_call=True,
)
def draw_violin_plot(selected_variable):
    if not selected_variable:
        return go.Figure()
    column = get_key_from_value(VARIABLE_DESCRIPTIONS, selected_variable)
    violin = make_violin_plot(df=data, col=column)
    return violin

@callback(
    Output("histplot", "figure"),
    [Input("variable-dropdown", "value")],
    prevent_initial_call=True,
)
def draw_hist_plot(selected_variable):
    if not selected_variable:
        return px.histogram()
    column = get_key_from_value(VARIABLE_DESCRIPTIONS, selected_variable)
    hist = make_hist(df=data, col=column)
    return hist
