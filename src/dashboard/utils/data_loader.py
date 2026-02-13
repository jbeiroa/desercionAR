import os
import pandas as pd
from pathlib import Path

# Define the base path to the data directory
# Using Path.home() to resolve the '~' correctly
DATA_PATH = "data/processed"
PREDICT_DATA_PATH = os.path.join(DATA_PATH, "predict.csv")
TRAIN_DATA_PATH = os.path.join(DATA_PATH, "train.csv")

def load_predict_data() -> pd.DataFrame:
    """
    Loads the prediction dataset.

    Returns:
        pd.DataFrame: The prediction data.
    """
    if not os.path.exists(PREDICT_DATA_PATH):
        raise FileNotFoundError(f"Predict data not found at {PREDICT_DATA_PATH}")
    return pd.read_csv(PREDICT_DATA_PATH)

def load_train_data() -> pd.DataFrame:
    """
    Loads the training dataset.

    Returns:
        pd.DataFrame: The training data.
    """
    if not os.path.exists(TRAIN_DATA_PATH):
        raise FileNotFoundError(f"Train data not found at {TRAIN_DATA_PATH}")
    return pd.read_csv(TRAIN_DATA_PATH).fillna(0)

def get_geojson_path() -> str:
    """
    Returns the path to the GeoJSON file.

    Returns:
        str: The path to the GeoJSON file.
    """
    return os.path.join(
        Path(__file__).parent.parent, "assets/Regiones.geojson"
    )

