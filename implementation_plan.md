# Implementation Plan - Healthcare AI Risk Prediction Platform

This plan outlines the design and implementation of a production-quality Healthcare AI Risk Prediction Platform from scratch, using the Diabetes 130-US hospitals dataset to predict 30-day hospital readmission risk.

## Goal Description

Build an end-to-end Machine Learning web application containing:
1. **Data Pipeline**: Automated data download, cleaning, feature engineering, and feature selection.
2. **Machine Learning Pipeline**: Multi-model training, hyperparameter tuning with GridSearchCV, and best model auto-selection.
3. **Explainable AI (XAI)**: SHAP & LIME explanation services.
4. **LLM Clinical Explainer**: Gemini-powered clinical insights, key risk factors summarization, and interactive chatbot for clinical questions.
5. **Backend**: FastAPI REST service with model serving, batch predictions, and retraining triggers.
6. **Frontend**: Modern, dark-themed Streamlit dashboard with interactive charts, patient search, risk prediction, and model comparison.
7. **Database**: SQLite tracking prediction history, uploads, and model metadata.
8. **Tracking & Logging**: MLflow tracking and robust python logging.
9. **DevOps & Verification**: Dockerized multi-container setup and comprehensive Pytest suite.

---

## Proposed Project Structure

We will implement the project matching the requested structure:

```
Healthcare-AI-Risk-Prediction/
├── app/
│   ├── backend/
│   │   ├── __init__.py
│   │   ├── main.py             # FastAPI entrypoint
│   │   ├── routes.py           # API endpoints (/predict, /batch-predict, /retrain, etc.)
│   │   └── schemas.py          # Pydantic request/response validation
│   ├── frontend/
│   │   ├── __init__.py
│   │   ├── main.py             # Streamlit entrypoint
│   │   ├── pages/              # Sub-pages (Data Explorer, Predict, Performance, XAI, Chat, Info)
│   │   └── styles.css          # Custom premium theme & styles
│   ├── models/
│   │   ├── __init__.py
│   │   ├── pipeline.py         # Full ML training & tuning pipeline
│   │   └── preprocessor.py     # Feature engineering & preprocessing class
│   ├── services/
│   │   ├── __init__.py
│   │   ├── db_service.py       # SQLite database operations
│   │   ├── xai_service.py      # SHAP & LIME explainers
│   │   ├── llm_service.py      # LLM clinical explanation & Q&A
│   │   └── report_service.py   # PDF Report generation
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── config_loader.py    # YAML configuration loader
│   │   ├── data_downloader.py  # Diabetes 130-US hospitals dataset downloader
│   │   └── logger.py           # Structured logger setup
│   └── config/
│       └── config.yaml         # Configuration parameters (paths, models, hyperparameters)
├── data/
│   ├── raw/                    # Downloaded raw dataset
│   └── processed/              # Cleaned & scaled dataset
├── notebooks/
│   └── eda.ipynb               # EDA scratchpad (optional documentation)
├── reports/
│   ├── preprocessed/           # Preprocessing & correlation reports
│   └── pdfs/                   # Generated patient risk PDFs
├── tests/
│   ├── __init__.py
│   ├── test_api.py             # API tests
│   ├── test_preprocessing.py   # Data cleaning & transformation tests
│   ├── test_training.py        # ML pipeline tests
│   └── test_utils.py           # Utilities tests
├── docs/
│   └── architecture.md         # Design documentation
├── models/
│   └── registry/               # Serialized model pickles & metadata
├── logs/
│   └── app.log                 # Log output file
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Detailed Component Plans

### 1. Configuration & Data Downloader (`config/config.yaml` & `utils/data_downloader.py`)
- **Config**: YAML file defining database paths, raw/processed data paths, model registry path, hyperparameter grids for tuning, MLflow tracking URI, and Gemini API settings.
- **Downloader**: Auto-downloads and extracts the UCI Diabetes 130-US hospitals dataset (ID 296) using python's `urllib` and `zipfile` from UCI's repository. Fallback instructions in logs and README if network errors occur.

### 2. Preprocessing & Feature Engineering (`models/preprocessor.py`)
- **Cleaning**: Replace "?" with NaN. Handle missing values (drop `weight`, `payer_code`, and `medical_specialty` if missing > 40%; impute `race`, `diag_1`, `diag_2`, `diag_3` with 'Unknown' or mode).
- **Medical Feature Engineering**: 
  - Group `diag_1`, `diag_2`, `diag_3` (ICD-9 codes) into clinical categories (Circulatory, Respiratory, Digestive, Diabetes, Injury, Musculoskeletal, Genitourinary, Neoplasms, and Other).
  - Create new features: `disease_severity_score` (combination of time in hospital, number of lab procedures, and number of medications) and `admission_frequency` (sum of emergency, inpatient, and outpatient visits in the previous year).
- **Preprocessing Pipeline**: One-hot encode categorical features, scale numerical features (StandardScaler), handle outliers via clipping/IQR, and select key features using Chi-Square/ANOVA or Random Forest feature importance.
- **Target Variable**: Map `readmitted` value `<30` to 1, and `>30`/`NO` to 0.

### 3. Model Training & Auto-Tuning Pipeline (`models/pipeline.py`)
- **Models**: Train Logistic Regression, Decision Tree, Random Forest, XGBoost, and Gradient Boosting. (Optional SVM and KNN can be integrated but configured with reasonable scaling to prevent slowdowns).
- **Tuning**: Implement `GridSearchCV` or `RandomizedSearchCV` on a cross-validation scheme.
- **Evaluation**: Log metrics to MLflow: Accuracy, Precision, Recall, F1, ROC-AUC. Plot ROC & Precision-Recall curves.
- **Auto-Selection**: Save the model with the highest validation metric (e.g. F1-score or ROC-AUC) to the model registry directory as `best_model.pkl` along with the preprocessor object.

### 4. Database Schema (`services/db_service.py`)
Use SQLite with tables:
- `predictions`:
  - `id` (INTEGER PRIMARY KEY)
  - `patient_id` (TEXT)
  - `features` (TEXT, JSON representation of input)
  - `prediction` (INTEGER)
  - `probability` (REAL)
  - `model_used` (TEXT)
  - `timestamp` (DATETIME)
  - `clinical_explanation` (TEXT)
- `model_metadata`:
  - `run_id` (TEXT PRIMARY KEY)
  - `model_name` (TEXT)
  - `metrics` (TEXT, JSON)
  - `hyperparameters` (TEXT, JSON)
  - `created_at` (DATETIME)
  - `is_active` (BOOLEAN)
- `uploads`:
  - `id` (INTEGER PRIMARY KEY)
  - `filename` (TEXT)
  - `row_count` (INTEGER)
  - `timestamp` (DATETIME)

### 5. Explainable AI & PDF Generation (`services/xai_service.py`, `services/report_service.py`)
- **XAI**: Implement SHAP TreeExplainer/KernelExplainer and LIME tabular explainer. Generate local SHAP force/waterfall values and global feature importance.
- **PDF Reports**: Use `ReportLab` to build professional, structured PDF reports detailing patient demographics, risk predictions, key SHAP/LIME contributors, and automated clinical recommendations.

### 6. LLM Clinical Explainer (`services/llm_service.py`)
- Check for `GEMINI_API_KEY` (env variable or config).
- If present, connect via Google GenAI or `google-generativeai` SDK.
- **Explain Prediction**: Combine raw patient stats + predicted probability + top SHAP features. Prompt the LLM to write a professional medical-style summary explaining *why* the patient is at risk, and actionable recommendations.
- **Chatbot & Q&A**: Maintain a message history. Answer user queries (e.g. "Why is this patient high risk despite having normal blood sugar?") using contextual prompt injection.
- **Mock Fallback**: If no API key is available, use a rule-based expert system that simulates high-quality medical explanations based on the feature values.

### 7. FastAPI Backend (`backend/main.py`, `backend/routes.py`, `backend/schemas.py`)
- **POST /predict**: Single inference. Saves prediction & explanation to SQLite database.
- **POST /batch-predict**: CSV upload or JSON list. Returns predictions, probabilities, and saves to database.
- **GET /model-info**: Returns details about active model, parameters, and training timestamp.
- **GET /metrics**: Retrieves training metrics from the latest run.
- **GET /health**: Service health status.
- **POST /retrain**: Asynchronously triggers the training pipeline to re-download or update model parameters.

### 8. Streamlit Frontend Dashboard (`frontend/main.py`)
- **Theme**: Dark theme, high-contrast, premium teal/blue/grey healthcare colors.
- **Pages**:
  1. **Home**: Overview of the platform, KPIs (total predictions, average risk, accuracy of current model).
  2. **Predict Patient Risk**: Form input for single patient + interactive SHAP/LIME explanation + clinical LLM recommendations + PDF download button.
  3. **Batch Predict**: File uploader for CSV files, progress bar, table of predictions, download links, and aggregate statistics.
  4. **Model Performance**: Compare all trained models. Interactive Plotly ROC, PR, and Confusion Matrix charts. Hyperparameter table.
  5. **SHAP Explainability**: Global explanations, feature importance plots, summary plots.
  6. **Data Explorer**: Interactive EDA charts (demographics, hospital stays, readmission breakdown, correlation heatmaps).
  7. **API Documentation**: Interactive OpenAPI explorer details or instructions.
  8. **About**: Details of the platform, tech stack, and portfolio-worthy descriptions.

### 9. DevOps & Testing (`Dockerfile`, `docker-compose.yml`, `tests/`)
- Multi-service Docker configuration:
  - Container 1: FastAPI Backend
  - Container 2: Streamlit Frontend
  - Container 3: MLflow tracking server (optional/local)
- Comprehensive test suite for utilities, preprocessing, pipeline, and API endpoints using `pytest` and `httpx`.

---

## Verification Plan

### Automated Tests
Run the test suite using `pytest`:
```bash
pytest tests/ -v
```

### Manual Verification
1. Run `python app/utils/data_downloader.py` to verify download and extraction.
2. Run `python app/models/pipeline.py` to verify full ML training, evaluation, auto-selection of the best model, and DB insert.
3. Start the FastAPI server: `uvicorn app.backend.main:app --reload --port 8000`
4. Start the Streamlit dashboard: `streamlit run app/frontend/main.py`
5. Test predicting a patient manually in the UI, verify the LLM explanation generation (using key or mock), and download the generated PDF.
6. Verify batch predictions using a test CSV.
7. Run and build Docker container via `docker-compose up --build`.
