import os
import sqlite3
import json
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
from app.utils.config_loader import load_config
from app.utils.logger import logger

def get_db_connection() -> sqlite3.Connection:
    """Returns a connection to the SQLite database, creating parent directories if necessary."""
    config = load_config()
    db_path = config["paths"]["database"]
    
    # Ensure directory exists
    db_dir = os.path.dirname(db_path)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir, exist_ok=True)
        
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def init_db() -> None:
    """Initializes the database schema."""
    logger.info("Initializing database...")
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Create predictions table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT,
            features TEXT, -- JSON string
            prediction INTEGER,
            probability REAL,
            model_used TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            clinical_explanation TEXT
        )
    """)
    
    # Create model_metadata table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_metadata (
            run_id TEXT PRIMARY KEY,
            model_name TEXT,
            metrics TEXT, -- JSON string
            hyperparameters TEXT, -- JSON string
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            is_active BOOLEAN DEFAULT 0
        )
    """)
    
    # Create uploads table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS uploads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            row_count INTEGER,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    conn.commit()
    conn.close()
    logger.info("Database initialized successfully.")

def save_prediction(
    patient_id: str,
    features: Dict[str, Any],
    prediction: int,
    probability: float,
    model_used: str,
    clinical_explanation: Optional[str] = None
) -> int:
    """Saves a patient risk prediction record to the database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute(
        """
        INSERT INTO predictions (patient_id, features, prediction, probability, model_used, clinical_explanation)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            patient_id,
            json.dumps(features),
            int(prediction),
            float(probability),
            model_used,
            clinical_explanation
        )
    )
    
    prediction_id = cursor.lastrowid
    conn.commit()
    conn.close()
    logger.info(f"Prediction saved successfully with ID {prediction_id} for Patient {patient_id}.")
    return prediction_id

def get_predictions_history(limit: int = 100) -> List[Dict[str, Any]]:
    """Retrieves a list of past predictions."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute(
        "SELECT id, patient_id, features, prediction, probability, model_used, timestamp, clinical_explanation "
        "FROM predictions ORDER BY timestamp DESC LIMIT ?",
        (limit,)
    )
    
    rows = cursor.fetchall()
    conn.close()
    
    history = []
    for row in rows:
        history.append({
            "id": row["id"],
            "patient_id": row["patient_id"],
            "features": json.loads(row["features"]) if row["features"] else {},
            "prediction": row["prediction"],
            "probability": row["probability"],
            "model_used": row["model_used"],
            "timestamp": row["timestamp"],
            "clinical_explanation": row["clinical_explanation"]
        })
        
    return history

def save_model_metadata(
    run_id: str,
    model_name: str,
    metrics: Dict[str, float],
    hyperparameters: Optional[Dict[str, Any]] = None,
    is_active: bool = False
) -> None:
    """Saves model training run metadata and sets active status."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    if is_active:
        # Deactivate all other models
        cursor.execute("UPDATE model_metadata SET is_active = 0")
        
    cursor.execute(
        """
        INSERT OR REPLACE INTO model_metadata (run_id, model_name, metrics, hyperparameters, is_active)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            run_id,
            model_name,
            json.dumps(metrics),
            json.dumps(hyperparameters or {}),
            1 if is_active else 0
        )
    )
    
    conn.commit()
    conn.close()
    logger.info(f"Model metadata saved for run {run_id} ({model_name}). Active={is_active}.")

def get_active_model_info() -> Optional[Dict[str, Any]]:
    """Retrieves information about the currently active model."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute(
        "SELECT run_id, model_name, metrics, hyperparameters, created_at "
        "FROM model_metadata WHERE is_active = 1 LIMIT 1"
    )
    
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return {
            "run_id": row["run_id"],
            "model_name": row["model_name"],
            "metrics": json.loads(row["metrics"]) if row["metrics"] else {},
            "hyperparameters": json.loads(row["hyperparameters"]) if row["hyperparameters"] else {},
            "created_at": row["created_at"]
        }
    return None

def save_upload(filename: str, row_count: int) -> int:
    """Logs a batch data file upload."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute(
        "INSERT INTO uploads (filename, row_count) VALUES (?, ?)",
        (filename, row_count)
    )
    
    upload_id = cursor.lastrowid
    conn.commit()
    conn.close()
    logger.info(f"Upload log saved with ID {upload_id} for file {filename}.")
    return upload_id
