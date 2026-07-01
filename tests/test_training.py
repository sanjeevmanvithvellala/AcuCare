import os
import json
import pandas as pd
import pytest
from unittest.mock import patch
from app.models.pipeline import train_and_evaluate_pipeline
from app.utils.config_loader import load_config

@pytest.fixture
def mock_dataset(tmp_path):
    """Creates a small mock patient dataset for fast training test execution."""
    data_dir = tmp_path / "data" / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    mock_data = pd.DataFrame({
        "encounter_id": list(range(1, 61)),
        "patient_nbr": list(range(1001, 1061)),
        "race": ["Caucasian", "AfricanAmerican", "Asian", "Other", "Hispanic"] * 12,
        "gender": ["Female", "Male"] * 30,
        "age": ["[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"] * 12,
        "time_in_hospital": [3, 5, 2, 8, 4] * 12,
        "num_lab_procedures": [40, 60, 30, 80, 45] * 12,
        "num_procedures": [1, 0, 2, 1, 0] * 12,
        "num_medications": [15, 22, 10, 35, 12] * 12,
        "number_outpatient": [0, 1, 0, 2, 0] * 12,
        "number_emergency": [0, 0, 1, 0, 0] * 12,
        "number_inpatient": [1, 0, 2, 0, 0] * 12,
        "number_diagnoses": [5, 7, 4, 9, 6] * 12,
        "diag_1": ["428", "250", "496", "577", "820"] * 12,
        "diag_2": ["250", "?", "428", "250", "?"] * 12,
        "diag_3": ["276", "?", "?", "?", "?",] * 12,
        "discharge_disposition_id": [1, 1, 2, 1, 1] * 12,
        "metformin": ["No", "Steady", "No", "No", "No"] * 12,
        "insulin": ["Steady", "No", "Up", "No", "Steady"] * 12,
        "change": ["No", "Ch", "Ch", "No", "No"] * 12,
        "diabetesMed": ["Yes", "Yes", "Yes", "No", "Yes"] * 12,
        "readmitted": ["<30", "NO", ">30", "NO", "<30"] * 12
    })
    
    csv_path = data_dir / "diabetic_data.csv"
    mock_data.to_csv(csv_path, index=False)
    return str(csv_path)

def test_pipeline_execution(mock_dataset, tmp_path):
    """Test full pipeline execution end-to-end using a mock configuration and dataset."""
    # Define temporary directories for test artifacts
    model_reg = tmp_path / "models" / "registry"
    proc_dir = tmp_path / "data" / "processed"
    rep_dir = tmp_path / "reports"
    
    model_reg.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)
    rep_dir.mkdir(parents=True, exist_ok=True)
    
    # Load default configuration and patch paths
    test_config = load_config()
    test_config["paths"]["raw_data_dir"] = os.path.dirname(mock_dataset)
    test_config["paths"]["model_registry"] = str(model_reg)
    test_config["paths"]["processed_data_dir"] = str(proc_dir)
    test_config["paths"]["reports_dir"] = str(rep_dir)
    test_config["paths"]["database"] = str(tmp_path / "test_db.db")
    test_config["mlflow"]["tracking_uri"] = f"sqlite:///{tmp_path}/mlflow.db"
    
    # Mock load_config to return our modified test_config
    # Mock download_dataset to skip download and return mock dataset file path
    with patch("app.models.pipeline.load_config", return_value=test_config), \
         patch("app.utils.data_downloader.download_dataset", return_value=mock_dataset):
         
        best_name, best_metric = train_and_evaluate_pipeline()
        
        # Verify output files were created
        assert best_name is not None
        assert best_metric >= 0.0
        
        # Check files
        assert os.path.exists(os.path.join(model_reg, "best_model.pkl"))
        assert os.path.exists(os.path.join(model_reg, "preprocessor.pkl"))
        assert os.path.exists(os.path.join(model_reg, "model_metadata.json"))
        assert os.path.exists(os.path.join(proc_dir, "train_processed.csv"))
        
        # Verify metadata contents
        with open(os.path.join(model_reg, "model_metadata.json"), "r") as f:
            meta = json.load(f)
            assert meta["model_name"] == best_name
            assert "metrics" in meta
            assert len(meta["features"]) > 0
            
        # Verify reports generated
        assert os.path.exists(os.path.join(rep_dir, "preprocessed", "correlation_matrix.csv"))
        assert os.path.exists(os.path.join(rep_dir, "preprocessed", "model_comparison.csv"))
        assert os.path.exists(os.path.join(rep_dir, "plots", "roc_comparison.png"))
        assert os.path.exists(os.path.join(rep_dir, "plots", "pr_comparison.png"))
        assert os.path.exists(os.path.join(rep_dir, "plots", "best_confusion_matrix.png"))
