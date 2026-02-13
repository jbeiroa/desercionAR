import dash
from dash import dcc, html, callback, Input, Output, State
import dash_bootstrap_components as dbc
import requests
import json
import pandas as pd

from dashboard.utils.translations import VARIABLE_DESCRIPTIONS

dash.register_page(__name__, path="/prediction")

# Features to include in the form
# These are based on the `importantes` list from the old home.py
form_features = [
    "CH06", "NIVEL_ED", "II8", "ratio_ocupados", "IX_TOT",
    "IX_MEN10", "CH08", "V12", "NIVEL_ED_jefx", "ESTADO",
]

# Create form inputs
form_inputs = [
    dbc.Row(
        [
            dbc.Label(VARIABLE_DESCRIPTIONS.get(feat, feat), width=4),
            dbc.Col(
                dcc.Input(id=f"input-{feat}", type="number", value=0),
                width=8,
            ),
        ],
        className="mb-3",
    )
    for feat in form_features
]

layout = dbc.Container(
    [
        html.H2("Single Prediction"),
        html.P("Fill the form to get a prediction for a single student."),
        dbc.Form(form_inputs),
        dbc.Button("Predict", id="predict-button-single", n_clicks=0, className="mt-3"),
        html.Div(id="prediction-output", className="mt-4"),
    ],
    fluid=True,
)

@callback(
    Output("prediction-output", "children"),
    Input("predict-button-single", "n_clicks"),
    [State(f"input-{feat}", "value") for feat in form_features],
    prevent_initial_call=True,
)
def get_prediction(n_clicks, *values):
    if n_clicks > 0:
        # Create the payload
        payload = {"data": [dict(zip(form_features, values))]}

        try:
            # Make the request to the API
            # Assuming the API is running on localhost:8000
            response = requests.post("http://localhost:8000/predict", json=payload)
            response.raise_for_status()  # Raise an exception for bad status codes
            result = response.json()

            # Display the result
            prediction = result["predictions"][0]["prediction"]
            probability = result["predictions"][0]["probability"]

            if prediction == 1:
                alert_color = "danger"
                prediction_text = "Dropout"
            else:
                alert_color = "success"
                prediction_text = "No Dropout"

            return dbc.Alert(
                f"Prediction: {prediction_text} (Probability: {probability:.2f})",
                color=alert_color,
            )

        except requests.exceptions.RequestException as e:
            return dbc.Alert(f"Error connecting to the API: {e}", color="danger")
        except Exception as e:
            return dbc.Alert(f"An error occurred: {e}", color="danger")

    return ""
