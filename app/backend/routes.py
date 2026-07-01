import os
import pandas as pd
import json
from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File, Depends
from typing import List, Dict, Any, Optional

from app.backend.schemas import (
    PatientInput, PredictionResponse, BatchPredictionResponse, ModelInfoResponse, HealthResponse
)
from app.utils.config_loader import load_config
from app.utils.logger import logger
from app.models.preprocessor import HealthcarePreprocessor
from app.models.pipeline import train_and_evaluate_pipeline
from app.services.xai_service import HealthcareXAIService
from app.services.llm_service import HealthcareLLMService
from app.services.db_service import (
    save_prediction, get_active_model_info, save_model_metadata, save_upload, init_db
)
import pickle

router = APIRouter()

# Global variables to store loaded models and services in memory
_model = None
_preprocessor = None
_xai_service = None
_llm_service = None

def load_ml_components():
    """Loads ML model, preprocessor, and XAI service into memory."""
    global _model, _preprocessor, _xai_service, _llm_service
    config = load_config()
    
    # Initialize LLM service (handles config and api key internally)
    _llm_service = HealthcareLLMService()
    
    registry_dir = config["paths"]["model_registry"]
    model_path = os.path.join(registry_dir, "best_model.pkl")
    preprocessor_path = os.path.join(registry_dir, "preprocessor.pkl")
    processed_train_path = os.path.join(config["paths"]["processed_data_dir"], "train_processed.csv")
    
    # 1. Load Preprocessor
    if not os.path.exists(preprocessor_path):
        logger.warning("Preprocessor not found. Retraining may be needed.")
        return False
        
    try:
        _preprocessor = HealthcarePreprocessor.load(preprocessor_path)
    except Exception as e:
        logger.error(f"Error loading preprocessor: {e}")
        return False
        
    # 2. Load Model
    if not os.path.exists(model_path):
        logger.warning("Best model pickle not found. Retraining may be needed.")
        return False
        
    try:
        with open(model_path, "rb") as f:
            _model = pickle.load(f)
        logger.info("Successfully loaded active machine learning model.")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False
        
    # 3. Load XAI Service (requires processed background data)
    if os.path.exists(processed_train_path):
        try:
            train_df = pd.read_csv(processed_train_path)
            # Separate features from target
            if "target" in train_df.columns:
                train_df = train_df.drop(columns=["target"])
            _xai_service = HealthcareXAIService(train_df, _preprocessor.selected_features)
            logger.info("Successfully initialized Explainable AI Service.")
        except Exception as e:
            logger.error(f"Error initializing XAI Service: {e}")
    else:
        logger.warning(f"Processed training file not found at {processed_train_path}. XAI disabled.")
        
    return True

# Retraining Background Task
def run_retrain_task():
    """Background task to retrain the ML pipeline."""
    try:
        logger.info("Asynchronous retraining task started...")
        best_name, best_f1 = train_and_evaluate_pipeline()
        
        # Load metadata
        config = load_config()
        registry_dir = config["paths"]["model_registry"]
        metadata_path = os.path.join(registry_dir, "model_metadata.json")
        
        with open(metadata_path, "r") as f:
            meta = json.load(f)
            
        # Generate a run ID if not set
        run_id = f"retrain_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Save to DB
        save_model_metadata(
            run_id=run_id,
            model_name=meta["model_name"],
            metrics=meta["metrics"],
            hyperparameters={},
            is_active=True
        )
        
        # Reload models into FastAPI memory
        success = load_ml_components()
        if success:
            logger.info("Retraining complete. New model successfully loaded into memory.")
        else:
            logger.error("Retraining complete, but failed to load new components into memory.")
    except Exception as e:
        logger.error(f"Retraining failed: {e}", exc_info=True)

# API Endpoints

@router.get("/health", response_model=HealthResponse)
def health_check():
    """Returns application health and system status."""
    db_connected = False
    try:
        # Simple test connection
        from app.services.db_service import get_db_connection
        conn = get_db_connection()
        conn.execute("SELECT 1")
        conn.close()
        db_connected = True
    except Exception as e:
        logger.error(f"Health check database error: {e}")
        
    model_loaded = _model is not None and _preprocessor is not None
    
    status = "healthy" if db_connected and model_loaded else "degraded"
    
    return HealthResponse(
        status=status,
        model_loaded=model_loaded,
        database_connected=db_connected
    )

@router.post("/predict", response_model=PredictionResponse)
def predict_patient(input_data: PatientInput):
    """Predicts 30-day readmission risk for a single patient."""
    global _model, _preprocessor, _xai_service, _llm_service
    
    if _model is None or _preprocessor is None:
        raise HTTPException(
            status_code=503, 
            detail="Machine learning model is not loaded. Please trigger retraining (/retrain) first."
        )
        
    try:
        # Convert Pydantic model to dictionary, allowing alias (dashes)
        patient_dict = input_data.model_dump(by_alias=True)
        
        # 1. Transform input features
        X_patient = _preprocessor.transform_single(patient_dict)
        
        # 2. Model Prediction
        prediction = int(_model.predict(X_patient)[0])
        probability = float(_model.predict_proba(X_patient)[0][1])
        risk_level = "High" if probability >= 0.5 else "Medium" if probability >= 0.2 else "Low"
        
        # 3. Explainability
        top_features = []
        shap_explanation = {"base_value": 0.5, "shap_values": []}
        
        if _xai_service is not None:
            shap_explanation = _xai_service.get_shap_explanation(_model, X_patient)
            top_features = _xai_service.get_top_risk_factors(shap_explanation, top_n=5)
            
        # 4. Clinical LLM Explanation
        clinical_explanation = "LLM Explainer is offline."
        if _llm_service is not None:
            clinical_explanation = _llm_service.generate_clinical_explanation(
                patient_raw=patient_dict,
                probability=probability,
                prediction=prediction,
                top_features=top_features
            )
            
        # 5. Save prediction in database
        model_name = type(_model).__name__
        save_prediction(
            patient_id=input_data.patient_id,
            features=patient_dict,
            prediction=prediction,
            probability=probability,
            model_used=model_name,
            clinical_explanation=clinical_explanation
        )
        
        return PredictionResponse(
            patient_id=input_data.patient_id,
            prediction=prediction,
            probability=probability,
            risk_level=risk_level,
            model_used=model_name,
            explanation=clinical_explanation,
            top_features=top_features
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@router.post("/batch-predict", response_model=BatchPredictionResponse)
def batch_predict(patients: List[PatientInput]):
    """Predicts readmission risk for a batch of patients submitted via JSON list."""
    if _model is None or _preprocessor is None:
        raise HTTPException(status_code=503, detail="ML model is not loaded.")
        
    predictions_list = []
    readmitted_count = 0
    
    for patient in patients:
        try:
            # Predict single
            res = predict_patient(patient)
            predictions_list.append(res)
            if res.prediction == 1:
                readmitted_count += 1
        except Exception as e:
            logger.warning(f"Failed prediction for batch record {patient.patient_id}: {e}")
            
    return BatchPredictionResponse(
        total_records=len(patients),
        readmitted_count=readmitted_count,
        predictions=predictions_list
    )

@router.post("/batch-predict-csv")
def batch_predict_csv(file: UploadFile = File(...)):
    """Predicts readmission risk from an uploaded CSV file, logs results, and returns prediction details."""
    global _model, _preprocessor
    if _model is None or _preprocessor is None:
        raise HTTPException(status_code=503, detail="ML model is not loaded.")
        
    try:
        # Read uploaded CSV
        df_upload = pd.read_csv(file.file)
        row_count = len(df_upload)
        
        # Save file upload meta in DB
        save_upload(file.filename, row_count)
        
        # Ensure it has a patient identifier or generate one
        if "patient_id" not in df_upload.columns and "patient_nbr" in df_upload.columns:
            df_upload["patient_id"] = df_upload["patient_nbr"].astype(str)
        elif "patient_id" not in df_upload.columns:
            df_upload["patient_id"] = [f"CSV_PT_{i}" for i in range(len(df_upload))]
            
        results = []
        readmitted_count = 0
        
        # Process rows
        for idx, row in df_upload.iterrows():
            patient_dict = row.to_dict()
            # Clean floats of NaN
            patient_dict = {k: (None if pd.isna(v) else v) for k, v in patient_dict.items()}
            
            patient_id = str(patient_dict.get("patient_id", f"CSV_{idx}"))
            
            # Map parameters to preprocessor
            try:
                # Transform features
                X_pat = _preprocessor.transform_single(patient_dict)
                
                # Predict
                prob = float(_model.predict_proba(X_pat)[0][1])
                pred = int(_model.predict(X_pat)[0])
                
                # SHAP Features (simplified for speed during batch processing)
                top_features = []
                if _xai_service is not None:
                    # Only do for first 10 rows to prevent massive slowing, otherwise mock
                    if idx < 10:
                        shap_explanation = _xai_service.get_shap_explanation(_model, X_pat)
                        top_features = _xai_service.get_top_risk_factors(shap_explanation, top_n=5)
                        
                # LLM Clinical interpretation (limit to first 3 to prevent rate limits / slowness)
                clinical_exp = "Batch prediction placeholder explanation."
                if idx < 3 and _llm_service is not None:
                    clinical_exp = _llm_service.generate_clinical_explanation(
                        patient_raw=patient_dict,
                        probability=prob,
                        prediction=pred,
                        top_features=top_features
                    )
                elif idx >= 3:
                    clinical_exp = f"Batch record prediction. Risk: {'High' if prob >= 0.5 else 'Low'}."
                    
                # Save to database
                model_name = type(_model).__name__
                save_prediction(
                    patient_id=patient_id,
                    features=patient_dict,
                    prediction=pred,
                    probability=prob,
                    model_used=model_name,
                    clinical_explanation=clinical_exp
                )
                
                results.append({
                    "patient_id": patient_id,
                    "prediction": pred,
                    "probability": prob,
                    "risk_level": "High" if prob >= 0.5 else "Medium" if prob >= 0.2 else "Low",
                    "model_used": model_name
                })
                
                if pred == 1:
                    readmitted_count += 1
            except Exception as row_err:
                logger.warning(f"Error predicting CSV row {idx}: {row_err}")
                
        return {
            "filename": file.filename,
            "total_records": row_count,
            "readmitted_count": readmitted_count,
            "predictions": results
        }
        
    except Exception as e:
        logger.error(f"Error processing batch CSV upload: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process CSV file: {str(e)}")

@router.get("/model-info", response_model=ModelInfoResponse)
def get_model_info():
    """Returns active model details, training parameters, and performance metrics."""
    info = get_active_model_info()
    if info is None:
        raise HTTPException(status_code=404, detail="No active model found in database metadata.")
        
    # Get features from preprocessor
    features = _preprocessor.selected_features if _preprocessor else []
    
    return ModelInfoResponse(
        model_name=info["model_name"],
        run_id=info["run_id"],
        metrics=info["metrics"],
        hyperparameters=info["hyperparameters"],
        created_at=info["created_at"],
        features=features
    )

@router.get("/metrics")
def get_model_metrics():
    """Retrieves comparative metrics of all models."""
    info = get_active_model_info()
    if info is None:
        return {"error": "No model has been trained yet."}
    return info["metrics"]

@router.post("/retrain")
def retrain_model(background_tasks: BackgroundTasks):
    """Triggers asynchronous retraining of the machine learning pipeline."""
    background_tasks.add_task(run_retrain_task)
    return {"message": "Retraining task triggered in background. System status will update upon completion."}

# Call onload to get things set up
try:
    load_ml_components()
except Exception as err:
    logger.warning(f"Startup ML loading warning: {err}. Retraining required.")
