import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.utils.logger import logger
from app.utils.config_loader import load_config
from app.services.db_service import init_db
from app.backend.routes import router, load_ml_components

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup actions
    logger.info("Initializing Healthcare AI Risk Prediction Backend Application...")
    
    # 1. Initialize SQLite database schema
    try:
        init_db()
    except Exception as db_err:
        logger.critical(f"Failed to initialize SQLite database: {db_err}")
        
    # 2. Load ML model, preprocessor and XAI explainers in memory
    try:
        load_ml_components()
    except Exception as ml_err:
        logger.warning(f"Could not load ML models on startup: {ml_err}. Run /retrain API or dashboard retraining.")
        
    yield
    
    # Shutdown actions
    logger.info("Shutting down Healthcare AI Risk Prediction Backend Application...")

config = load_config()

app = FastAPI(
    title=config["project"]["name"],
    version=config["project"]["version"],
    description="A production-ready platform that predicts 30-day hospital readmission risk using explainable AI (SHAP & LIME) and LLM-powered clinical interpretation.",
    lifespan=lifespan
)

# Configure CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this to specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API Router
app.include_router(router)

if __name__ == "__main__":
    uvicorn.run("app.backend.main:app", host="0.0.0.0", port=8000, reload=True)
