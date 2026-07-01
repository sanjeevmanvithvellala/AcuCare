from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class PatientInput(BaseModel):
    patient_id: str = Field(..., description="Unique Patient Identifier")
    race: Optional[str] = Field("Unknown", description="Patient Race Category")
    gender: Optional[str] = Field("Unknown", description="Patient Gender")
    age: Optional[str] = Field("[50-60)", description="Age range, e.g., '[50-60)'")
    time_in_hospital: Optional[int] = Field(3, description="Number of days in hospital")
    num_lab_procedures: Optional[int] = Field(40, description="Number of lab procedures performed")
    num_procedures: Optional[int] = Field(1, description="Number of non-lab procedures performed")
    num_medications: Optional[int] = Field(15, description="Number of medications administered")
    number_outpatient: Optional[int] = Field(0, description="Number of outpatient visits in previous year")
    number_emergency: Optional[int] = Field(0, description="Number of emergency visits in previous year")
    number_inpatient: Optional[int] = Field(0, description="Number of inpatient visits in previous year")
    number_diagnoses: Optional[int] = Field(5, description="Number of diagnoses listed on record")
    
    diag_1: Optional[str] = Field("250", description="Primary ICD-9 diagnosis code")
    diag_2: Optional[str] = Field("Unknown", description="Secondary ICD-9 diagnosis code")
    diag_3: Optional[str] = Field("Unknown", description="Tertiary ICD-9 diagnosis code")
    
    discharge_disposition_id: Optional[int] = Field(1, description="Discharge disposition ID code")
    admission_type_id: Optional[int] = Field(1, description="Admission type ID code")
    admission_source_id: Optional[int] = Field(7, description="Admission source ID code")
    
    # Medications
    metformin: Optional[str] = "No"
    repaglinide: Optional[str] = "No"
    nateglinide: Optional[str] = "No"
    chlorpropamide: Optional[str] = "No"
    glimepiride: Optional[str] = "No"
    acetohexamide: Optional[str] = "No"
    glipizide: Optional[str] = "No"
    glyburide: Optional[str] = "No"
    tolbutamide: Optional[str] = "No"
    pioglitazone: Optional[str] = "No"
    rosiglitazone: Optional[str] = "No"
    acarbose: Optional[str] = "No"
    miglitol: Optional[str] = "No"
    troglitazone: Optional[str] = "No"
    tolazamide: Optional[str] = "No"
    examide: Optional[str] = "No"
    citoglipton: Optional[str] = "No"
    insulin: Optional[str] = "No"
    
    # Handle hyphens using alias and allow population by name/alias
    glyburide_metformin: Optional[str] = Field("No", alias="glyburide-metformin")
    glipizide_metformin: Optional[str] = Field("No", alias="glipizide-metformin")
    glimepiride_pioglitazone: Optional[str] = Field("No", alias="glimepiride-pioglitazone")
    metformin_rosiglitazone: Optional[str] = Field("No", alias="metformin-rosiglitazone")
    metformin_pioglitazone: Optional[str] = Field("No", alias="metformin-pioglitazone")
    
    change: Optional[str] = "No"
    diabetesMed: Optional[str] = "No"
    
    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "patient_id": "PT99482",
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
                "discharge_disposition_id": 1,
                "admission_type_id": 1,
                "admission_source_id": 7,
                "metformin": "Steady",
                "insulin": "Up",
                "change": "Ch",
                "diabetesMed": "Yes"
            }
        }

class PredictionResponse(BaseModel):
    patient_id: str
    prediction: int
    probability: float
    risk_level: str
    model_used: str
    explanation: Optional[str] = None
    top_features: List[Dict[str, Any]]
    
class BatchPredictionResponse(BaseModel):
    total_records: int
    readmitted_count: int
    predictions: List[PredictionResponse]

class ModelInfoResponse(BaseModel):
    model_name: str
    run_id: str
    metrics: Dict[str, float]
    hyperparameters: Dict[str, Any]
    created_at: str
    features: List[str]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    database_connected: bool
