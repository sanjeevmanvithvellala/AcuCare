import os
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# Import app components for testing
from app.backend.main import app
from app.backend.routes import load_ml_components
import app.backend.routes as routes

client = TestClient(app)

@pytest.fixture(autouse=True)
def mock_db_and_ml():
    """Autouse fixture to mock database connection and ML model load during tests."""
    with patch("app.backend.routes.save_prediction", return_value=1), \
         patch("app.backend.routes.get_active_model_info") as mock_info, \
         patch("app.services.db_service.get_db_connection"):
         
        # Setup mock active model metadata
        mock_info.return_value = {
            "model_name": "MockClassifier",
            "run_id": "mock_run_123",
            "metrics": {"accuracy": 0.85, "f1_score": 0.72},
            "hyperparameters": {},
            "created_at": "2026-06-30 18:00:00"
        }
        
        # Instantiate a mock model & preprocessor in routes global state
        mock_model = MagicMock()
        mock_model.predict.return_value = [1]
        mock_model.predict_proba.return_value = [[0.2, 0.8]]
        type(mock_model).__name__ = "MockClassifier"
        
        mock_preprocessor = MagicMock()
        mock_preprocessor.selected_features = ["time_in_hospital", "num_lab_procedures"]
        mock_preprocessor.transform_single.return_value = MagicMock()
        
        mock_xai = MagicMock()
        mock_xai.get_shap_explanation.return_value = {
            "base_value": 0.5,
            "shap_values": [
                {"feature": "time_in_hospital", "value": 4.0, "shap_value": 0.25},
                {"feature": "num_lab_procedures", "value": 45.0, "shap_value": 0.15}
            ]
        }
        mock_xai.get_top_risk_factors.return_value = [
            {"feature": "time_in_hospital", "value": 4.0, "shap_value": 0.25}
        ]
        
        # Inject mocks
        routes._model = mock_model
        routes._preprocessor = mock_preprocessor
        routes._xai_service = mock_xai
        
        yield

def test_health_endpoint():
    """Verify the /health endpoint status and status flags."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["healthy", "degraded"]
    assert data["model_loaded"] is True

def test_model_info_endpoint():
    """Verify /model-info returns active model metadata and feature list."""
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "MockClassifier"
    assert "time_in_hospital" in data["features"]

def test_predict_single_patient():
    """Verify single patient prediction workflow, schemas and responses."""
    payload = {
        "patient_id": "PT-TEST-01",
        "race": "Caucasian",
        "gender": "Male",
        "age": "[60-70)",
        "time_in_hospital": 4,
        "num_lab_procedures": 48,
        "num_procedures": 2,
        "num_medications": 18,
        "number_outpatient": 0,
        "number_emergency": 1,
        "number_inpatient": 1,
        "number_diagnoses": 6,
        "diag_1": "428",
        "diag_2": "250",
        "diag_3": "276",
        "discharge_disposition_id": 1
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["patient_id"] == "PT-TEST-01"
    assert data["prediction"] == 1
    assert data["probability"] == 0.8
    assert data["risk_level"] == "High"
    assert "explanation" in data
    assert len(data["top_features"]) > 0

def test_batch_predict():
    """Verify batch prediction JSON endpoint schema and output totals."""
    payload = [
        {
            "patient_id": "PT-BATCH-1",
            "time_in_hospital": 3,
            "num_lab_procedures": 30
        },
        {
            "patient_id": "PT-BATCH-2",
            "time_in_hospital": 5,
            "num_lab_procedures": 50
        }
    ]
    
    response = client.post("/batch-predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_records"] == 2
    assert len(data["predictions"]) == 2
    assert data["predictions"][0]["patient_id"] == "PT-BATCH-1"
