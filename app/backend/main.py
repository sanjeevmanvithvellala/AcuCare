import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.utils.logger import logger
from app.utils.config_loader import load_config
from app.services.db_service import init_db
from app.backend.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup and shutdown events.
    Keep startup lightweight to reduce Render memory usage.
    """

    logger.info("Starting Healthcare AI Risk Prediction Backend...")

    # Initialize SQLite database only
    try:
        init_db()
        logger.info("Database initialized successfully.")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")

    logger.info("ML components will be loaded lazily on first prediction request.")

    yield

    logger.info("Shutting down Healthcare AI Risk Prediction Backend...")


config = load_config()

app = FastAPI(
    title=config["project"]["name"],
    version=config["project"]["version"],
    description="Healthcare AI Risk Prediction Platform",
    lifespan=lifespan,
)

# -----------------------------
# CORS
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Restrict in production if desired
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Register API Routes
# -----------------------------
app.include_router(router)


@app.get("/")
def root():
    """
    Root endpoint for Render health checks.
    """
    return {
        "status": "running",
        "service": "Healthcare AI Risk Prediction API",
        "docs": "/docs"
    }


if __name__ == "__main__":
    uvicorn.run(
        "app.backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )
