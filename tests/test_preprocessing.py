import pandas as pd
import numpy as np
from app.models.preprocessor import HealthcarePreprocessor

def test_icd9_mapping():
    """Verify that ICD-9 codes are mapped to clinical groups correctly."""
    # Circulatory: 390-459 or 785
    assert HealthcarePreprocessor.map_icd9("428") == "Circulatory"
    assert HealthcarePreprocessor.map_icd9("785.51") == "Circulatory"
    
    # Diabetes: 250
    assert HealthcarePreprocessor.map_icd9("250.02") == "Diabetes"
    
    # Respiratory: 460-519 or 786
    assert HealthcarePreprocessor.map_icd9("496") == "Respiratory"
    
    # Other: invalid or unmapped codes
    assert HealthcarePreprocessor.map_icd9("V58.61") == "Other"
    assert HealthcarePreprocessor.map_icd9("?") == "Other"
    assert HealthcarePreprocessor.map_icd9(np.nan) == "Other"

def test_clean_data_filters_deceased():
    """Verify that patients discharged to hospice or who died are filtered out."""
    preprocessor = HealthcarePreprocessor()
    
    # discharge_disposition_id 11 indicates expired (died), 1 is home
    mock_data = pd.DataFrame({
        "encounter_id": [1, 2, 3],
        "patient_nbr": [101, 102, 103],
        "discharge_disposition_id": [1, 11, 13],  # 11, 13 should be filtered
        "readmitted": ["NO", "<30", ">30"]
    })
    
    cleaned = preprocessor.clean_data(mock_data, is_training=True)
    
    # Only the first patient (discharged to home) should remain
    assert len(cleaned) == 1
    assert cleaned.iloc[0]["encounter_id"] == 1

def test_fit_and_transform_pipeline():
    """Test full preprocessing fit & transform on mock dataset."""
    preprocessor = HealthcarePreprocessor(target_col="readmitted")
    
    # Create mock patient DataFrame
    mock_train = pd.DataFrame({
        "encounter_id": [1, 2, 3, 4, 5],
        "patient_nbr": [101, 102, 103, 104, 105],
        "race": ["Caucasian", "AfricanAmerican", "Asian", "Caucasian", "Hispanic"],
        "gender": ["Female", "Male", "Female", "Male", "Female"],
        "age": ["[50-60)", "[60-70)", "[70-80)", "[50-60)", "[80-90)"],
        "time_in_hospital": [3, 5, 2, 8, 4],
        "num_lab_procedures": [40, 60, 30, 80, 45],
        "num_procedures": [1, 0, 2, 1, 0],
        "num_medications": [15, 22, 10, 35, 12],
        "number_outpatient": [0, 1, 0, 2, 0],
        "number_emergency": [0, 0, 1, 0, 0],
        "number_inpatient": [1, 0, 2, 0, 0],
        "number_diagnoses": [5, 7, 4, 9, 6],
        "diag_1": ["428", "250", "496", "577", "820"],
        "diag_2": ["250", "Unknown", "428", "250", "Unknown"],
        "diag_3": ["276", "Unknown", "Unknown", "Unknown", "Unknown"],
        "discharge_disposition_id": [1, 1, 2, 1, 1],
        "metformin": ["No", "Steady", "No", "No", "No"],
        "insulin": ["Steady", "No", "Up", "No", "Steady"],
        "change": ["No", "Ch", "Ch", "No", "No"],
        "diabetesMed": ["Yes", "Yes", "Yes", "No", "Yes"],
        "readmitted": ["<30", "NO", ">30", "NO", "<30"]
    })
    
    preprocessor.fit(mock_train)
    
    # Verify features were fitted and scaler is ready
    assert len(preprocessor.selected_features) > 0
    assert hasattr(preprocessor.scaler, "mean_")
    
    # Transform mock data
    X_trans, y_trans = preprocessor.transform(mock_train)
    
    # Check transformed shape
    assert X_trans.shape[0] == len(mock_train)
    assert X_trans.shape[1] == len(preprocessor.selected_features)
    assert len(y_trans) == len(mock_train)
    
    # Check single transformation dictionary
    patient_dict = mock_train.iloc[0].to_dict()
    X_single = preprocessor.transform_single(patient_dict)
    assert X_single.shape == (1, len(preprocessor.selected_features))
