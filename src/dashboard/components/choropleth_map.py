import json
import pandas as pd
import plotly.express as px
from dash import dcc, html, callback, Input, Output
from src.dashboard.utils.data_loader import get_geojson_path

def choropleth_map(data: pd.DataFrame, predictions: pd.Series) -> html.Div:
    """
    Creates the choropleth map component.

    Args:
        data (pd.DataFrame): The data for the map.
        predictions (pd.Series): The model's predictions.

    Returns:
        html.Div: The choropleth map component.
    """
    with open(get_geojson_path()) as regs:
        regiones = json.load(regs)

    @callback(Output("choropleth-map", "figure"), [Input("predict-button", "n_clicks")])
    def update_choropleth(n_clicks):
        if n_clicks is None:
            group = pd.DataFrame(data.groupby("REGION").size()).reset_index()
            group.rename({"REGION": "CodigoEPH", 0: "personas"}, axis=1, inplace=True)
            fig = px.choropleth(
                group,
                geojson=regiones,
                locations="CodigoEPH",
                color="personas",
                color_continuous_scale="Viridis",
                range_color=(group.personas.min(), group.personas.max()),
                scope="south america",
                projection="mercator",
                featureidkey="properties.CodigoEPH",
            )
            fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
            return fig

        data["DESERTO"] = predictions
        group = pd.DataFrame(data[data.DESERTO == 1].groupby("REGION").size().reset_index())
        group.rename({"REGION": "CodigoEPH", 0: "desertores"}, axis=1, inplace=True)
        fig = px.choropleth(
            group,
            geojson=regiones,
            locations="CodigoEPH",
            color="desertores",
            color_continuous_scale="Viridis",
            range_color=(group.desertores.min(), group.desertores.max()),
            scope="south america",
            projection="mercator",
            featureidkey="properties.CodigoEPH",
        )
        fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
        return fig

    return html.Div(
        [
            dcc.Graph(id="choropleth-map"),
        ]
    )
