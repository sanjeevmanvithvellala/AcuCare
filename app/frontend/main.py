import os
import sys

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json

from app.utils.config_loader import load_config
from app.services.report_service import generate_pdf_report
from app.services.llm_service import HealthcareLLMService

# Setup page layout
st.set_page_config(
    page_title="Healthcare AI Risk Prediction Platform",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Backend URL (default)
BACKEND_URL = os.environ.get(
    "BACKEND_URL",
    "https://acucare.onrender.com"
)

# Load CSS styles
def load_css():
    css_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "styles.css")
    if os.path.exists(css_file):
        with open(css_file, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# Cache raw dataset for EDA
@st.cache_data
def load_raw_data(filepath: str):
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        df = df.replace('?', np.nan)
        return df
    return None

# Load config
config = load_config()
raw_data_path = os.path.join(config["paths"]["raw_data_dir"], config["data"]["raw_filename"])
df_raw = load_raw_data(raw_data_path)

# Sidebar Navigation
st.sidebar.markdown(
    "<div style='text-align: center; padding: 10px;'>"
    "<h2 style='color: #2a9d8f; margin-bottom: 0;'>🏥 AcuCare AI</h2>"
    "<p style='color: #888; font-size: 12px; margin-top: 0;'>Predictive Readmission Platform</p>"
    "</div>",
    unsafe_allow_html=True
)

st.sidebar.markdown("---")

page = st.sidebar.selectbox(
    "Navigation Menu",
    [
        "Home", 
        "Predict Patient Risk", 
        "Batch Predict", 
        "Model Performance", 
        "SHAP Explainability", 
        "Data Explorer", 
        "API Documentation", 
        "About"
    ]
)

st.sidebar.markdown("---")

# Health Status in Sidebar
try:
    health_response = requests.get(f"{BACKEND_URL}/health", timeout=3)
    if health_response.status_code == 200:
        health_data = health_response.json()
        if health_data["status"] == "healthy":
            st.sidebar.success("🟢 API Status: Connected")
        else:
            st.sidebar.warning("🟡 API Status: Degraded")
    else:
        st.sidebar.error("🔴 API Status: Error")
except Exception:
    st.sidebar.error("🔴 API Status: Disconnected")

# -----------------
# PAGE 1: HOME
# -----------------
if page == "Home":
    st.title("🏥 Healthcare AI Risk Prediction Platform")
    st.markdown(
        "Welcome to the **CareRisk AI** platform, an enterprise-grade clinical decision support tool "
        "designed to predict the 30-day readmission risk of diabetic patients. The system utilizes "
        "explainable Machine Learning (SHAP & LIME) paired with generative LLM clinical reasoning "
        "to assist care teams in optimizing discharge coordination and patient outcomes."
    )
    
    st.markdown("### 📊 System Dashboard Overview")
    
    # Render KPI metrics
    col1, col2, col3, col4 = st.columns(4)
    
    # Fetch active model details
    active_model = "No Active Model"
    active_auc = 0.0
    active_f1 = 0.0
    model_timestamp = "N/A"
    
    try:
        model_info_resp = requests.get(f"{BACKEND_URL}/model-info", timeout=3)
        if model_info_resp.status_code == 200:
            info = model_info_resp.json()
            active_model = info["model_name"]
            active_auc = info["metrics"].get("roc_auc", 0.0)
            active_f1 = info["metrics"].get("f1_score", 0.0)
            model_timestamp = info.get("created_at", "N/A")
    except Exception:
        pass
        
    with col1:
        st.markdown(
            f"<div class='kpi-card'>"
            f"<div class='kpi-title'>Active Model</div>"
            f"<div class='kpi-value'>{active_model.replace('Classifier', '').replace('_', ' ')}</div>"
            f"</div>",
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            f"<div class='kpi-card'>"
            f"<div class='kpi-title'>Model F1-Score</div>"
            f"<div class='kpi-value'>{active_f1:.4f}</div>"
            f"</div>",
            unsafe_allow_html=True
        )
    with col3:
        st.markdown(
            f"<div class='kpi-card'>"
            f"<div class='kpi-title'>Model ROC-AUC</div>"
            f"<div class='kpi-value'>{active_auc:.4f}</div>"
            f"</div>",
            unsafe_allow_html=True
        )
    with col4:
        total_records = len(df_raw) if df_raw is not None else 0
        st.markdown(
            f"<div class='kpi-card'>"
            f"<div class='kpi-title'>Base Patient Registry</div>"
            f"<div class='kpi-value'>{total_records:,}</div>"
            f"</div>",
            unsafe_allow_html=True
        )
        
    st.markdown("### 🔄 Machine Learning Pipeline Control")
    st.markdown(
        "Click the button below to execute the full training, hyperparameter tuning, model comparison, "
        "and auto-selection pipeline. Results are logged to MLflow."
    )
    
    if st.button("🚀 Trigger ML Pipeline Retraining"):
        with st.spinner("Training pipeline running asynchronously in background..."):
            try:
                resp = requests.post(f"{BACKEND_URL}/retrain")
                if resp.status_code == 200:
                    st.success("Training successfully triggered! Please wait 1-2 minutes for completion. Check the console or logs for details.")
                else:
                    st.error(f"Failed to trigger retraining: {resp.text}")
            except Exception as e:
                st.error(f"Error connecting to backend: {e}")
                
    st.markdown("---")
    st.markdown("### 🛠️ Architecture Workflow")
    st.markdown(
        "1. **Ingestion & Preprocessing**: Cleans raw data, drops dead/hospice cases, imputes categories, and maps ICD-9 codes to categories.\n"
        "2. **Auto-Selection**: Compares Logistic Regression, Decision Tree, Random Forest, Calibrated SVM, KNN, and XGBoost.\n"
        "3. **Inference & Explainability**: Predicts individual outcomes via REST API. Calculates SHAP feature impact and LIME local explanation.\n"
        "4. **Clinical Interpretation**: Translates features and statistics into clinical summaries and guidelines using an LLM."
    )

# -----------------
# PAGE 2: PREDICT PATIENT RISK
# -----------------
elif page == "Predict Patient Risk":
    st.title("🏥 Predict Single Patient Readmission Risk")
    st.markdown("Enter patient demographic and clinical details to generate readmission probability and explanations.")
    
    st.markdown("### 📝 Patient Details Form")
    
    # Forms
    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)
        
        with c1:
            patient_id = st.text_input("Patient ID", value="PT-88391")
            race = st.selectbox("Race", ["Caucasian", "AfricanAmerican", "Asian", "Hispanic", "Other", "Unknown"])
            gender = st.selectbox("Gender", ["Female", "Male"])
            age = st.selectbox("Age Range", ["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)", "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"], index=6)
            
        with c2:
            time_in_hospital = st.slider("Time in Hospital (Days)", 1, 14, 4)
            num_lab_procedures = st.number_input("Number of Lab Procedures", min_value=1, max_value=150, value=45)
            num_procedures = st.number_input("Number of Non-Lab Procedures", min_value=0, max_value=10, value=1)
            num_medications = st.number_input("Number of Medications", min_value=1, max_value=100, value=16)
            
        with c3:
            number_inpatient = st.number_input("Previous Inpatient Admissions (Year)", min_value=0, max_value=20, value=0)
            number_emergency = st.number_input("Previous Emergency Room Visits (Year)", min_value=0, max_value=20, value=0)
            number_outpatient = st.number_input("Previous Outpatient Visits (Year)", min_value=0, max_value=20, value=0)
            number_diagnoses = st.number_input("Number of Diagnoses", min_value=1, max_value=20, value=7)
            
        st.markdown("#### Diagnosis & Medications")
        c4, c5, c6 = st.columns(3)
        
        with c4:
            diag_1 = st.text_input("Primary Diagnosis (ICD-9)", value="428") # Congestive Heart Failure
            diag_2 = st.text_input("Secondary Diagnosis (ICD-9)", value="250") # Diabetes
            diag_3 = st.text_input("Tertiary Diagnosis (ICD-9)", value="276") # Hyperosmolality
            
        with c5:
            insulin = st.selectbox("Insulin Dosage Change", ["No", "Steady", "Up", "Down"])
            metformin = st.selectbox("Metformin Dosage Change", ["No", "Steady", "Up", "Down"])
            
        with c6:
            change = st.selectbox("Any Diabetes Med Change?", ["No", "Ch"])
            diabetesMed = st.selectbox("On Diabetes Medication?", ["No", "Yes"], index=1)
            discharge_disposition_id = st.number_input("Discharge Disposition Code (e.g. 1 = Home)", min_value=1, max_value=30, value=1)
            
        submit_btn = st.form_submit_button("🩺 Predict Risk & Explain")
        
    if submit_btn:
        # Construct JSON request payload
        payload = {
            "patient_id": patient_id,
            "race": race,
            "gender": gender,
            "age": age,
            "time_in_hospital": int(time_in_hospital),
            "num_lab_procedures": int(num_lab_procedures),
            "num_procedures": int(num_procedures),
            "num_medications": int(num_medications),
            "number_outpatient": int(number_outpatient),
            "number_emergency": int(number_emergency),
            "number_inpatient": int(number_inpatient),
            "number_diagnoses": int(number_diagnoses),
            "diag_1": diag_1,
            "diag_2": diag_2,
            "diag_3": diag_3,
            "metformin": metformin,
            "insulin": insulin,
            "change": change,
            "diabetesMed": diabetesMed,
            "discharge_disposition_id": int(discharge_disposition_id)
        }
        
        with st.spinner("Analyzing patient metrics and generating explanation..."):
            try:
                resp = requests.post(f"{BACKEND_URL}/predict", json=payload)
                if resp.status_code == 200:
                    result = resp.json()
                    
                    st.success("Analysis Complete!")
                    
                    # Risk Classification Alert
                    prob = result["probability"]
                    risk_lvl = result["risk_level"]
                    
                    risk_class = "risk-high" if prob >= 0.5 else "risk-medium" if prob >= 0.2 else "risk-low"
                    
                    st.markdown(
                        f"<div class='kpi-card {risk_class}'>"
                        f"<div style='font-size: 16px; font-weight: 600;'>Patient Classification: {risk_lvl} Risk</div>"
                        f"<div style='font-size: 32px; font-weight: 700; margin-top: 5px;'>{prob * 100:.1f}% Probability</div>"
                        f"<div style='font-size: 12px; color: #888; margin-top: 5px;'>Model Evaluated: {result['model_used']}</div>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                    
                    # Two column view: SHAP chart on left, Clinical report on right
                    col_left, col_right = st.columns([1, 1])
                    
                    with col_left:
                        st.markdown("### 📊 Top Risk Feature Impacts (SHAP)")
                        
                        top_features = result["top_features"]
                        if top_features:
                            feat_names = [f["feature"].replace("_", " ").title() for f in top_features]
                            feat_vals = [f["shap_value"] for f in top_features]
                            
                            # Create Plotly Horizontal Bar Chart
                            fig = go.Figure(go.Bar(
                                x=feat_vals,
                                y=feat_names,
                                orientation='h',
                                marker_color=['#ef4444' if x > 0 else '#10b981' for x in feat_vals]
                            ))
                            fig.update_layout(
                                xaxis_title="SHAP Value (Risk Contribution)",
                                yaxis=dict(autorange="reversed"),
                                margin=dict(l=20, r=20, t=20, b=20),
                                height=350,
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                font_color="#c9d1d9"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("No SHAP values returned from API. Feature selection may have restricted columns.")
                            
                    with col_right:
                        st.markdown("### 📝 Clinical Explanation & Guidance")
                        st.markdown(result["explanation"])
                        
                        # Generate PDF Button
                        try:
                            # We can call the generator on current machine
                            pdf_path = generate_pdf_report(
                                patient_id=patient_id,
                                patient_raw=payload,
                                probability=prob,
                                prediction=result["prediction"],
                                top_features=top_features,
                                model_name=result["model_used"],
                                clinical_explanation=result["explanation"]
                            )
                            
                            with open(pdf_path, "rb") as f:
                                st.download_button(
                                    label="📄 Download PDF Risk Report",
                                    data=f,
                                    file_name=os.path.basename(pdf_path),
                                    mime="application/pdf"
                                )
                        except Exception as pdf_err:
                            st.error(f"Error generating PDF file: {pdf_err}")
                            
                    # Chatbot Section
                    st.markdown("---")
                    st.markdown("### 💬 Ask Questions About Patient Risk")
                    st.markdown("Interact with the clinical AI chatbot to drill down into the risk assessment.")
                    
                    if "chat_history" not in st.session_state:
                        st.session_state["chat_history"] = []
                        
                    # Display history
                    for chat in st.session_state["chat_history"]:
                        if chat["role"] == "user":
                            st.markdown(f"**👤 Clinician:** {chat['content']}")
                        else:
                            st.markdown(f"**🤖 Clinical AI:** {chat['content']}")
                            
                    user_q = st.text_input("Ask a question about the prediction (e.g., 'How does this patient's age and insulin affect their risk?')", key="chat_input")
                    
                    if st.button("Send Query"):
                        if user_q:
                            # Append user message
                            st.session_state["chat_history"].append({"role": "user", "content": user_q})
                            
                            # Fetch chatbot response
                            llm_svc = HealthcareLLMService()
                            chatbot_resp = llm_svc.chat_about_patient(
                                patient_raw=payload,
                                probability=prob,
                                top_features=top_features,
                                chat_history=st.session_state["chat_history"][:-1],
                                user_message=user_q
                            )
                            
                            # Append bot message and rerun
                            st.session_state["chat_history"].append({"role": "model", "content": chatbot_resp})
                            st.rerun()
                            
                else:
                    st.error(f"Prediction API Error: {resp.text}")
            except Exception as conn_err:
                st.error(f"Failed to communicate with API server: {conn_err}")

# -----------------
# PAGE 3: BATCH PREDICT
# -----------------
elif page == "Batch Predict":
    st.title("📂 Upload Batch Patient Records")
    st.markdown("Upload a CSV file containing multiple patient profiles to run parallel risk predictions.")
    
    # Download template CSV
    st.markdown("#### 📥 CSV Format Template")
    st.markdown("Ensure your uploaded CSV contains matching column headers. Click below to download a template.")
    
    # Generate quick template
    template_cols = ["patient_id", "race", "gender", "age", "time_in_hospital", "num_lab_procedures", 
                     "num_procedures", "num_medications", "number_outpatient", "number_emergency", 
                     "number_inpatient", "number_diagnoses", "diag_1", "diag_2", "diag_3", 
                     "metformin", "insulin", "change", "diabetesMed", "discharge_disposition_id"]
    template_row = ["PT_99011", "Caucasian", "Male", "[50-60)", 3, 35, 1, 12, 0, 0, 1, 6, "250", "428", "Other", "No", "Steady", "No", "Yes", 1]
    template_df = pd.DataFrame([template_row], columns=template_cols)
    
    template_csv = template_df.to_csv(index=False)
    st.download_button("Download CSV Template", template_csv, "readmission_template.csv", "text/csv")
    
    st.markdown("---")
    st.markdown("#### 📤 Upload File")
    uploaded_file = st.file_uploader("Select patient CSV file", type=["csv"])
    
    if uploaded_file is not None:
        st.success(f"File uploaded: {uploaded_file.name}")
        
        if st.button("🩺 Process Batch Predictions"):
            with st.spinner("Processing records..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
                    resp = requests.post(f"{BACKEND_URL}/batch-predict-csv", files=files)
                    
                    if resp.status_code == 200:
                        batch_res = resp.json()
                        st.success("Batch Prediction Completed!")
                        
                        # Display stats
                        tot = batch_res["total_records"]
                        readmit = batch_res["readmitted_count"]
                        rate = (readmit / tot) * 100 if tot > 0 else 0
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Records Processed", tot)
                        with col2:
                            st.metric("Predicted Readmissions (<30 days)", readmit)
                        with col3:
                            st.metric("Expected Readmission Rate", f"{rate:.1f}%")
                            
                        # Show output DataFrame
                        preds_df = pd.DataFrame(batch_res["predictions"])
                        st.dataframe(preds_df)
                        
                        # Download predictions
                        out_csv = preds_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Batch Predictions CSV",
                            data=out_csv,
                            file_name=f"predictions_{uploaded_file.name}",
                            mime="text/csv"
                        )
                    else:
                        st.error(f"Batch prediction failed: {resp.text}")
                except Exception as e:
                    st.error(f"Error connecting to backend API: {e}")

# -----------------
# PAGE 4: MODEL PERFORMANCE
# -----------------
elif page == "Model Performance":
    st.title("📈 Model Performance & Comparisons")
    st.markdown(
        "Compare metric evaluations across the trained machine learning classifiers. "
        "The model showing the highest F1-Score or ROC-AUC is automatically deployed to the API registry."
    )
    
    # Try reading the reports
    reports_dir = config["paths"]["reports_dir"]
    comparison_csv = os.path.join(reports_dir, "preprocessed", "model_comparison.csv")
    
    if os.path.exists(comparison_csv):
        comp_df = pd.read_csv(comparison_csv, index_index=False) if "index" in pd.read_csv(comparison_csv).columns else pd.read_csv(comparison_csv)
        comp_df = comp_df.rename(columns={"Unnamed: 0": "Model"})
        
        st.markdown("### 📊 Metrics Comparison Table")
        st.dataframe(comp_df.style.highlight_max(subset=["accuracy", "precision", "recall", "f1_score", "roc_auc"], color="#0d5c75"))
        
        # Display bar chart comparison of F1 and ROC AUC
        st.markdown("### 📊 Graphical Metrics Comparison")
        fig_comp = px.bar(
            comp_df,
            x="Model",
            y=["f1_score", "roc_auc"],
            barmode="group",
            labels={"value": "Score", "variable": "Metric"},
            color_discrete_sequence=["#2a9d8f", "#0f4c81"]
        )
        fig_comp.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="#c9d1d9")
        st.plotly_chart(fig_comp, use_container_width=True)
        
        # Display ROC and PR Curves if they exist
        st.markdown("### 📉 Evaluated Prediction Curves")
        c1, c2 = st.columns(2)
        
        roc_plot = os.path.join(reports_dir, "plots", "roc_comparison.png")
        pr_plot = os.path.join(reports_dir, "plots", "pr_comparison.png")
        
        with c1:
            if os.path.exists(roc_plot):
                st.image(roc_plot, caption="ROC Curves Comparison")
            else:
                st.info("ROC comparison plot not found.")
                
        with c2:
            if os.path.exists(pr_plot):
                st.image(pr_plot, caption="Precision-Recall Curves Comparison")
            else:
                st.info("PR comparison plot not found.")
                
        # Best model confusion matrix
        st.markdown("### 🔲 Best Model Confusion Matrix")
        cm_plot = os.path.join(reports_dir, "plots", "best_confusion_matrix.png")
        if os.path.exists(cm_plot):
            st.image(cm_plot, caption="Confusion Matrix - Best Selected Model", width=500)
    else:
        st.warning("No performance records found. Please trigger model retraining on the Home page to train classifiers.")

# -----------------
# PAGE 5: SHAP EXPLAINABILITY
# -----------------
elif page == "SHAP Explainability":
    st.title("🧠 Global Explainable AI (XAI) Analysis")
    st.markdown("Understand what features globally drive predictions across the entire patient population using SHAP values.")
    
    # Display details
    st.markdown("### 🔍 Feature Importance")
    st.markdown(
        "SHAP values measure the average impact of a feature on the final readmission probability. "
        "Positive values push predictions towards high readmission risk, while negative values reduce it."
    )
    
    active_info = None
    try:
        model_info_resp = requests.get(f"{BACKEND_URL}/model-info", timeout=3)
        if model_info_resp.status_code == 200:
            active_info = model_info_resp.json()
    except Exception:
        pass
        
    if active_info:
        st.info(f"Currently active model for explanation: **{active_info['model_name']}**")
        
        # Display global importance bar plot from the reports/plots directory
        # Since we ran preprocessor feature selection, we can construct feature names
        # Let's read comparison or model metadata
        registry_dir = config["paths"]["model_registry"]
        meta_path = os.path.join(registry_dir, "model_metadata.json")
        
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
                
            features = meta.get("features", [])
            st.markdown("#### Selected Features used in current Model:")
            st.write(features)
            
            st.markdown("---")
            st.markdown("#### Interpretability Frameworks (SHAP / LIME)")
            st.markdown(
                "- **SHAP (SHapley Additive exPlanations)**: Globally evaluates the relative contribution of each feature to predictions across the entire population. Locally explains how much each feature pushes probability above/below base values.\n"
                "- **LIME (Local Interpretable Model-agnostic Explanations)**: Identifies local decision boundaries for individual predictions, expressing coefficients for the specific inputs."
            )
        else:
            st.warning("Model metadata file not found.")
    else:
        st.warning("No active model is registered. Please trigger model retraining on the Home page.")

# -----------------
# PAGE 6: DATA EXPLORER
# -----------------
elif page == "Data Explorer":
    st.title("🗃️ Patient Cohort Data Explorer (EDA)")
    st.markdown("Explore demographics and clinical attributes of the Diabetes 130-US hospitals cohort.")
    
    if df_raw is not None:
        st.markdown(f"Total patient encounters: **{len(df_raw):,}**")
        
        # EDA tabs
        tab1, tab2, tab3 = st.tabs(["Demographics", "Clinical Distribution", "Target Distribution"])
        
        with tab1:
            st.markdown("#### Demographic Characteristics")
            col_d1, col_d2 = st.columns(2)
            
            with col_d1:
                # Race
                race_counts = df_raw["race"].value_counts(dropna=False).reset_index()
                fig_race = px.pie(race_counts, values="count", names="race", title="Patient Race Breakdown", color_discrete_sequence=px.colors.qualitative.Pastel)
                fig_race.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color="#c9d1d9")
                st.plotly_chart(fig_race, use_container_width=True)
                
            with col_d2:
                # Age distribution
                age_counts = df_raw["age"].value_counts().sort_index().reset_index()
                fig_age = px.bar(age_counts, x="age", y="count", title="Patient Age Distribution", color_discrete_sequence=["#0f4c81"])
                fig_age.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="#c9d1d9")
                st.plotly_chart(fig_age, use_container_width=True)
                
        with tab2:
            st.markdown("#### Clinical Feature Distributions")
            feat = st.selectbox(
                "Select clinical feature to view distribution:",
                ["time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications", "number_diagnoses"]
            )
            
            fig_hist = px.histogram(
                df_raw, 
                x=feat, 
                color="readmitted",
                barmode="overlay",
                title=f"Distribution of {feat.replace('_', ' ').title()} by Readmission Status",
                color_discrete_map={"<30": "#ef4444", ">30": "#f4a261", "NO": "#2a9d8f"}
            )
            fig_hist.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="#c9d1d9")
            st.plotly_chart(fig_hist, use_container_width=True)
            
        with tab3:
            st.markdown("#### Target variable (Readmission)")
            readmit_counts = df_raw["readmitted"].value_counts().reset_index()
            fig_target = px.bar(
                readmit_counts,
                x="readmitted",
                y="count",
                title="30-day Readmission Class Counts",
                color="readmitted",
                color_discrete_map={"<30": "#ef4444", ">30": "#f4a261", "NO": "#2a9d8f"}
            )
            fig_target.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="#c9d1d9")
            st.plotly_chart(fig_target, use_container_width=True)
            
    else:
        st.error("No dataset found. Please trigger dataset download in pipeline or download manually.")

# -----------------
# PAGE 7: API DOCUMENTATION
# -----------------
elif page == "API Documentation":
    st.title("🔌 API Documentation & Integration Guide")
    st.markdown("Integrate readmission predictions directly into Electronic Health Record (EHR) systems via REST API.")
    
    st.markdown("### 📍 API Swagger UI Endpoint")
    st.markdown(f"Access interactive REST documentation at: [**{BACKEND_URL}/docs**]({BACKEND_URL}/docs)")
    
    st.markdown("---")
    st.markdown("### 💻 Integration Code Examples")
    
    st.markdown("#### 1. Python Request")
    st.code(
        """import requests\n\n"""
        """url = "http://localhost:8000/predict"\n"""
        """payload = {\n"""
        """    "patient_id": "PT123",\n"""
        """    "race": "Caucasian",\n"""
        """    "gender": "Female",\n"""
        """    "age": "[60-70)",\n"""
        """    "time_in_hospital": 3,\n"""
        """    "num_lab_procedures": 40,\n"""
        """    "num_procedures": 1,\n"""
        """    "num_medications": 14,\n"""
        """    "number_outpatient": 0,\n"""
        """    "number_emergency": 0,\n"""
        """    "number_inpatient": 0,\n"""
        """    "number_diagnoses": 5,\n"""
        """    "diag_1": "428",\n"""
        """    "diag_2": "250",\n"""
        """    "diag_3": "276",\n"""
        """    "metformin": "No",\n"""
        """    "insulin": "Steady",\n"""
        """    "change": "No",\n"""
        """    "diabetesMed": "Yes",\n"""
        """    "discharge_disposition_id": 1\n"""
        """}\n\n"""
        """response = requests.post(url, json=payload)\n"""
        """print(response.json())\n""",
        language="python"
    )
    
    st.markdown("#### 2. cURL Request")
    st.code(
        """curl -X POST "http://localhost:8000/predict" \\\n"""
        """     -H "Content-Type: application/json" \\\n"""
        """     -d '{\n"""
        """       "patient_id": "PT123",\n"""
        """       "race": "Caucasian",\n"""
        """       "gender": "Female",\n"""
        """       "age": "[60-70)",\n"""
        """       "time_in_hospital": 3,\n"""
        """       "num_lab_procedures": 40,\n"""
        """       "num_procedures": 1,\n"""
        """       "num_medications": 14,\n"""
        """       "number_outpatient": 0,\n"""
        """       "number_emergency": 0,\n"""
        """       "number_inpatient": 0,\n"""
        """       "number_diagnoses": 5,\n"""
        """       "diag_1": "428",\n"""
        """       "diag_2": "250",\n"""
        """       "diag_3": "276",\n"""
        """       "metformin": "No",\n"""
        """       "insulin": "Steady",\n"""
        """       "change": "No",\n"""
        """       "diabetesMed": "Yes",\n"""
        """       "discharge_disposition_id": 1\n"""
        """     }'""",
        language="bash"
    )

# -----------------
# PAGE 8: ABOUT
# -----------------
elif page == "About":
    st.title("ℹ️ About the CareRisk AI Platform")
    st.markdown(
        "This platform represents a **Production-Quality Healthcare AI Risk Prediction Platform** "
        "built to demonstrate software engineering and machine learning best practices.\n\n"
        "It is designed to align with portfolio requirements for senior AI/ML engineer roles at firms like "
        "Acuitas360, Optum, Oracle Health, Microsoft, and Google."
    )
    
    st.markdown("### 📋 Developer Notes & Tech Stack")
    st.markdown(
        "- **Backend**: FastAPI REST service with complete type hints and Pydantic validation schemas.\n"
        "- **Frontend**: Streamlit dynamic dashboard with high contrast styling, interactive Plotly charting, and conversational chatbot UI.\n"
        "- **Machine Learning**: Standard scikit-learn models, XGBoost, and hyperparameter tuning pipeline.\n"
        "- **Explainable AI (XAI)**: Dual framework explainers (SHAP + LIME) mapping local and global feature influences.\n"
        "- **Clinical LLM**: Generates clinical explanations using the Gemini API, falling back to an expert rule-based engine when API credentials are not provided.\n"
        "- **Reporting**: Automated ReportLab PDF generation formatting patient risk and action guidelines.\n"
        "- **Database**: SQLite tracking metadata, file uploads, and inference history.\n"
        "- **Docker Support**: Containerized architecture for simple multi-container local deployments."
    )
