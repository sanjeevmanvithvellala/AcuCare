# Walkthrough - Healthcare AI Risk Prediction Platform

The platform is fully implemented with a complete, production-ready python codebase containing preprocessing, training, serving, visualization, explainability, logging, database persistence, unit testing, and Docker orchestration components.

## What Was Built

### 1. Core Configuration & Logger
- [config.yaml](file:///c:/Users/sanje/Desktop/Healthcare%20AI%20Risk%20Prediction%20Platform/app/config/config.yaml): Centralized configuration managing file paths, features to drop, dataset defaults, MLflow database setups, LLM prompts, and hyperparameter tuning grids.
- [logger.py](file:///c:/Users/sanje/Desktop/Healthcare%20AI%20Risk%20Prediction%20Platform/app/utils/logger.py) & [config_loader.py](file:///c:/Users/sanje/Desktop/Healthcare%20AI%20Risk%20Prediction%20Platform/app/utils/config_loader.py): Structured rotating file and console logging + absolute path resolver.

### 2. Preprocessing & Feature Engineering
- [preprocessor.py](file:///c:/Users/sanje/Desktop/Healthcare%20AI%20Risk%20Prediction%20Platform/app/models/preprocessor.py):
  - **Clinical Domain Filters**: Drops records of deceased or hospice patients (disposition codes 11, 13, 14, 19, 20, 21) since they cannot be readmitted.
  - **ICD-9 Diagnosis Mapper**: Groups raw ICD-9 codes into clinical categories (Circulatory, Respiratory, Digestive, Diabetes, Injury, Musculoskeletal, Genitourinary, Neoplasms, and Other).
  - **Custom Severity Score**: Combines index time in hospital, lab tests, and medications.
  - **Visit Frequency Score**: Merges inpatient, outpatient, and emergency encounters.
  - **Data Cleansing**: Handles missing markers ("?"), maps target categories, scales numerical features, and selects features via ANOVA SelectKBest.

### 3. Machine Learning Pipeline & Hyperparameter Tuning
- [pipeline.py](file:///c:/Users/sanje/Desktop/Healthcare%20AI%20Risk%20Prediction%20Platform/app/models/pipeline.py):
  - Compares: Logistic Regression, Decision Tree, Random Forest, Calibrated Linear SVM, KNN, XGBoost, and Gradient Boosting.
  - Automatically tunes models using `GridSearchCV` or `RandomizedSearchCV` on a stratifed split.
  - Generates evaluation reports: ROC and PR comparison curves, confusion matrix, and performance metrics CSVs saved under `reports/`.
  - Automatically selects the model with the highest validation F1-score and serializes it in the model registry.

### 4. Persistence & Audit Trails
- [db_service.py](file:///c:/Users/sanje/Desktop/Healthcare%20AI%20Risk%20Prediction%20Platform/app/services/db_service.py):
  - Configures SQLite schema on startup.
  - Logs patient prediction history (demographics, input features, probabilities, clinical explanations).
  - Stores model metadata, hyperparameter structures, and upload logs.

### 5. Explainable AI & PDF Generation
- [xai_service.py](file:///c:/Users/sanje/Desktop/Healthcare%20AI%20Risk%20Prediction%20Platform/app/services/xai_service.py): Wraps **SHAP Tree/Linear/Kernel Explainer** and **LIME Tabular Explainer** to measure local feature weights.
- [report_service.py](file:///c:/Users/sanje/Desktop/Healthcare%20AI%20Risk%20Prediction%20Platform/app/services/report_service.py): Generates structured clinical risk report PDFs using ReportLab.

### 6. LLM Clinical Explainer
- [llm_service.py](file:///c:/Users/sanje/Desktop/Healthcare%20AI%20Risk%20Prediction%20Platform/app/services/llm_service.py): Integrated **Gemini API** using `google-generativeai`. Fallback to an **Expert Clinician Rule-Based Engine** is automatically activated if `GEMINI_API_KEY` is not present, guaranteeing out-of-the-box functionality. Also includes a clinical conversational chatbot.

### 7. FastAPI Backend
- [main.py](file:///c:/Users/sanje/Desktop/Healthcare%20AI%20Risk%20Prediction%20Platform/app/backend/main.py), [routes.py](file:///c:/Users/sanje/Desktop/Healthcare%20AI%20Risk%20Prediction%20Platform/app/backend/routes.py), & [schemas.py](file:///c:/Users/sanje/Desktop/Healthcare%20AI%20Risk%20Prediction%20Platform/app/backend/schemas.py): Exposes endpoints (`/predict`, `/batch-predict`, `/batch-predict-csv`, `/model-info`, `/metrics`, `/health`, `/retrain`) with full Pydantic request/response validations.

### 8. Streamlit Frontend Dashboard
- [main.py](file:///c:/Users/sanje/Desktop/Healthcare%20AI%20Risk%20Prediction%20Platform/app/frontend/main.py): Renders the 8-page dashboard.
- [styles.css](file:///c:/Users/sanje/Desktop/Healthcare%20AI%20Risk%20Prediction%20Platform/app/frontend/styles.css): Injects a modern navy/teal healthcare theme.

### 9. DevOps & Docker Setup
- [Dockerfile](file:///c:/Users/sanje/Desktop/Healthcare%20AI%20Risk%20Prediction%20Platform/Dockerfile) & [docker-compose.yml](file:///c:/Users/sanje/Desktop/Healthcare%20AI%20Risk%20Prediction%20Platform/docker-compose.yml): Configures multi-container docker environments with persistent database/report volume maps and endpoint health checks.

---

## What Was Tested

We wrote a robust unit and integration testing suite under `tests/` that runs offline, fast, and does not require active internet connections:
1. [test_utils.py](file:///c:/Users/sanje/Desktop/Healthcare%20AI%20Risk%20Prediction%20Platform/tests/test_utils.py): Validates YAML config loader, relative-to-absolute path resolution, and log files creation.
2. [test_preprocessing.py](file:///c:/Users/sanje/Desktop/Healthcare%20AI%20Risk%20Prediction%20Platform/tests/test_preprocessing.py): Validates ICD-9 categories mapping and dead/hospice record filtering.
3. [test_training.py](file:///c:/Users/sanje/Desktop/Healthcare%20AI%20Risk%20Prediction%20Platform/tests/test_training.py): Sets up a mock patient dataset of 60 records, mocks raw downloading, and executes the entire ML pipeline (training, comparisons, metric curves, model registries, metadata dumps).
4. [test_api.py](file:///c:/Users/sanje/Desktop/Healthcare%20AI%20Risk%20Prediction%20Platform/tests/test_api.py): Spawns FastAPI `TestClient`, mocks prediction/DB hooks, and checks `/health`, `/model-info`, `/predict` (single), and `/batch-predict` endpoints.
