import dash
from dash import html
import dash_bootstrap_components as dbc
import mlflow
from mlflow.exceptions import MlflowException

from dashboard.components.choropleth_map import choropleth_map
from dashboard.components.feature_plots import feature_plots
from dashboard.utils.data_loader import load_predict_data
from dashboard.utils.translations import HOME_PRESENTATION

dash.register_page(__name__, path="/")

# Load data and model
data = load_predict_data()
model_uri = "models:/dropout_predictor/latest"
try:
    model = mlflow.pyfunc.load_model(model_uri)
except MlflowException as e:
    print(f"Error loading model: {e}")
    model = None

# Make predictions
id_cols = ["CODUSU", "NRO_HOGAR", "COMPONENTE", "ANO4", "TRIMESTRE", "DESERTO"]
X = data.loc[:, ~data.columns.isin(id_cols)].dropna()
predictions = model.predict(X) if model else None

layout = dbc.Container(
    [
        dbc.Row([dbc.Container(children=HOME_PRESENTATION, id="texto-principal")]),
        dbc.Row(
            [
                dbc.Col(
                    html.Div(
                        [
                            dbc.Button(
                                "Predict",
                                color="secondary",
                                className="me-1",
                                id="predict-button",
                            ),
                            choropleth_map(data, predictions),
                            dbc.Container(
                                html.Img(
                                    src="assets/conf_matrix.png", style={"width": "85%"}
                                )
                            ),
                        ],
                        style={"position": "sticky", "top": 0},
                    ),
                    width=4,
                ),
                dbc.Col(feature_plots(data, predictions), width=8),
            ]
        ),
    ],
    fluid=True,
)


