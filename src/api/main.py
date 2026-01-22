"""Main module for the FastAPI application.

This module defines the FastAPI application and its endpoints.
"""

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import mlflow
import numpy as np


# Define the Pydantic models
class PredictionInput(BaseModel):
    """
    Pydantic model for the input data for prediction.
    It accepts a list of records, where each record is a dictionary of feature names and values.
    """

    data: list[dict]


class PredictionOutput(BaseModel):
    """
    Pydantic model for the prediction output.
    It returns a list of predictions, where each prediction is a dictionary containing the prediction and the prediction probability.
    """

    predictions: list[dict]


class Health(BaseModel):
    """
    Pydantic model for the health check endpoint.
    """

    status: str


# Initialize the FastAPI app
app = FastAPI(
    title="School Dropout Prediction API",
    description="API to predict school dropout using the best model from the MLflow Model Registry.",
    version="0.1.0",
)

# Load the model from the MLflow Model Registry
model_uri = "models:/dropout_predictor/latest"
try:
    model = mlflow.pyfunc.load_model(model_uri)
except mlflow.exceptions.MlflowException as e:
    # This is a workaround for the case where the model is not found.
    # In a real-world scenario, you would want to handle this more gracefully.
    print(f"Error loading model: {e}")
    model = None


@app.get("/health", response_model=Health)
def health():
    """
    Health check endpoint.
    """
    return Health(status="ok")


@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: PredictionInput):
    """
    Prediction endpoint.
    Takes a list of records as input and returns a list of predictions.
    """
    if model is None:
        return PredictionOutput(predictions=[{"error": "Model not available"}])

    # Convert the input data to a pandas DataFrame
    input_df = pd.DataFrame(input_data.data)

    # Remove identifier columns that are not features
    id_cols_to_drop = [
        "CODUSU",
        "NRO_HOGAR",
        "ANO4",
        "COMPONENTE",
        "DESERTO",
        "TRIMESTRE",
    ]
    input_df = input_df.drop(columns=id_cols_to_drop, errors="ignore")

    # Make predictions
    try:
        predictions = model.predict(input_df)
    except Exception as e:
        return PredictionOutput(predictions=[{"error": str(e)}])

    # Get prediction probabilities
    try:
        # Check if the model has a `predict_proba` method
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df)
            # Assuming binary classification, get the probability of the positive class
            positive_class_prob = probabilities[:, 1]
        else:
            # If not, we can't provide a probability
            positive_class_prob = np.full(len(predictions), -1.0)
    except Exception as e:
        # If predict_proba fails, we can't provide a probability
        positive_class_prob = np.full(len(predictions), -1.0)

    # Format the output
    output = []
    for i in range(len(predictions)):
        output.append(
            {
                "prediction": int(predictions[i]),
                "probability": float(positive_class_prob[i]),
            }
        )

    return PredictionOutput(predictions=output)
