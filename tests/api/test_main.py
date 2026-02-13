import importlib
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import pytest
import numpy as np
import api.main
import mlflow

@pytest.fixture
def mock_model():
    """Fixture to create a mock MLflow model."""
    mock = MagicMock()
    # The mock should have a `predict` method that accepts a DataFrame
    mock.predict = MagicMock(return_value=np.array([1, 0]))
    # It should also have a `predict_proba` method
    mock.predict_proba = MagicMock(return_value=np.array([[0.1, 0.9], [0.8, 0.2]]))
    return mock

def test_health_endpoint():
    """Test the /health endpoint."""
    client = TestClient(api.main.app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_predict_success(mock_model):
    """Test the /predict endpoint with a successful prediction."""
    with patch('mlflow.pyfunc.load_model', return_value=mock_model):
        importlib.reload(api.main)
        client = TestClient(api.main.app)

        payload = {
            "data": [
                {"feature1": 0.5, "feature2": 0.3},
                {"feature1": 0.2, "feature2": 0.8}
            ]
        }
        response = client.post("/predict", json=payload)
        
        assert response.status_code == 200
        predictions = response.json()["predictions"]
        assert len(predictions) == 2
        
        assert predictions[0]["prediction"] == 1
        assert pytest.approx(predictions[0]["probability"], 0.01) == 0.9
        
        assert predictions[1]["prediction"] == 0
        assert pytest.approx(predictions[1]["probability"], 0.01) == 0.2

def test_predict_invalid_payload():
    """Test the /predict endpoint with an invalid payload."""
    client = TestClient(api.main.app)
    payload = {"invalid_key": "some_value"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Unprocessable Entity

def test_predict_model_not_available():
    """Test the /predict endpoint when the model is not available."""
    with patch('mlflow.pyfunc.load_model', side_effect=mlflow.exceptions.MlflowException("Model not found")):
        importlib.reload(api.main)
        client = TestClient(api.main.app)
        
        payload = {
            "data": [{"feature1": 0.5, "feature2": 0.3}]
        }
        response = client.post("/predict", json=payload)
        
        assert response.status_code == 200
        assert response.json() == {"predictions": [{"error": "Model not available"}]}

def test_predict_prediction_error(mock_model):
    """Test the /predict endpoint when model.predict raises an exception."""
    mock_model.predict.side_effect = Exception("Prediction failed")
    with patch('mlflow.pyfunc.load_model', return_value=mock_model):
        importlib.reload(api.main)
        client = TestClient(api.main.app)

        payload = {
            "data": [{"feature1": 0.5, "feature2": 0.3}]
        }
        response = client.post("/predict", json=payload)
        
        assert response.status_code == 200
        assert response.json() == {"predictions": [{"error": "Prediction failed"}]}

def test_predict_no_predict_proba(mock_model):
    """Test the /predict endpoint with a model that doesn't have predict_proba."""
    mock_model.predict.return_value = np.array([1])
    # To simulate no predict_proba, we delete it from the mock
    del mock_model.predict_proba
    
    with patch('mlflow.pyfunc.load_model', return_value=mock_model):
        importlib.reload(api.main)
        client = TestClient(api.main.app)
    
        payload = {
            "data": [{"feature1": 0.5, "feature2": 0.3}]
        }
        response = client.post("/predict", json=payload)
        
        assert response.status_code == 200
        predictions = response.json()["predictions"]
        assert predictions[0]["prediction"] == 1
        assert predictions[0]["probability"] == -1.0

def test_predict_predict_proba_error(mock_model):
    """Test the /predict endpoint when model.predict_proba raises an exception."""
    mock_model.predict.return_value = np.array([1])
    mock_model.predict_proba.side_effect = Exception("Proba failed")
    
    with patch('mlflow.pyfunc.load_model', return_value=mock_model):
        importlib.reload(api.main)
        client = TestClient(api.main.app)
    
        payload = {
            "data": [{"feature1": 0.5, "feature2": 0.3}]
        }
        response = client.post("/predict", json=payload)
        
        assert response.status_code == 200
        predictions = response.json()["predictions"]
        assert predictions[0]["prediction"] == 1
        assert predictions[0]["probability"] == -1.0
