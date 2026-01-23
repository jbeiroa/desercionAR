import pandas as pd
import plotly.express as px
from dash import dcc, html, callback, Input, Output
import dash_bootstrap_components as dbc
from src.dashboard.utils.translations import get_variable_description

def feature_plots(data: pd.DataFrame, predictions: pd.Series) -> html.Div:
    """
    Creates the feature plots component.

    Args:
        data (pd.DataFrame): The data for the plots.
        predictions (pd.Series): The model's predictions.

    Returns:
        html.Div: The feature plots component.
    """
    importantes = [
        "CH06", "NIVEL_ED", "II8", "ratio_ocupados", "IX_TOT",
        "IX_MEN10", "CH08", "V12", "NIVEL_ED_jefx", "ESTADO",
    ]

    @callback(
        [Output(f"chart{i}", "figure") for i in range(1, 11)],
        [Input("predict-button", "n_clicks")],
    )
    def update_charts(n_clicks):
        if n_clicks is None:
            charts = []
            for i in range(1, 11):
                charts.append(
                    px.histogram(
                        data_frame=data,
                        x=importantes[i - 1],
                        color="CH04",
                        labels={"CH04": "Gender"},
                        title=get_variable_description(importantes[i - 1]),
                    )
                )
            return charts
        
        data["DESERTO"] = predictions
        charts = []
        for i in range(1, 11):
            charts.append(
                px.histogram(
                    data_frame=data,
                    x=importantes[i - 1],
                    color="DESERTO",
                    title=get_variable_description(importantes[i - 1]),
                )
            )
        return charts

    return html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(
                        html.Div(
                            [
                                html.H4(get_variable_description(importantes[i - 1])),
                                dcc.Graph(id=f"chart{i}"),
                            ]
                        ),
                        width=4,
                    )
                    for i in range(1, 4)
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        html.Div(
                            [
                                html.H4(get_variable_description(importantes[i - 1])),
                                dcc.Graph(id=f"chart{i}"),
                            ]
                        ),
                        width=4,
                    )
                    for i in range(4, 7)
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        html.Div(
                            [
                                html.H4(get_variable_description(importantes[i - 1])),
                                dcc.Graph(id=f"chart{i}"),
                            ]
                        ),
                        width=4,
                    )
                    for i in range(7, 10)
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        html.Div(
                            [
                                html.H4(get_variable_description(importantes[9])),
                                dcc.Graph(id="chart10"),
                            ]
                        ),
                        width=4,
                    )
                ]
            ),
        ]
    )