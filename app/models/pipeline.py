import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve
)
# Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb

import mlflow
import mlflow.sklearn

from app.utils.logger import logger
from app.utils.config_loader import load_config
from app.utils.data_downloader import download_dataset
from app.models.preprocessor import HealthcarePreprocessor

def train_and_evaluate_pipeline():
    """Executes the full training, hyperparameter tuning, model comparison, and registry pipeline."""
    # 1. Load Configuration
    config = load_config()
    random_state = config["project"]["random_state"]
    
    # 2. Download/Locate Dataset
    raw_data_path = download_dataset()
    logger.info(f"Loading raw dataset from {raw_data_path}...")
    df_raw = pd.read_csv(raw_data_path)
    
    # 3. Create splits
    train_df, test_df = train_test_split(
        df_raw, 
        test_size=config["data"]["test_size"], 
        random_state=random_state,
        stratify=df_raw['readmitted'] if 'readmitted' in df_raw.columns else None
    )
    logger.info(f"Train split size: {train_df.shape}, Test split size: {test_df.shape}")
    
    # 4. Initialize and fit preprocessor
    preprocessor = HealthcarePreprocessor(
        target_col=config["data"]["target_col"],
        random_state=random_state
    )
    preprocessor.fit(train_df)
    
    # Transform splits
    X_train, y_train = preprocessor.transform(train_df)
    X_test, y_test = preprocessor.transform(test_df)
    
    logger.info(f"Transformed features size: {X_train.shape}")
    
    # Save Preprocessor
    preprocessor_path = os.path.join(config["paths"]["model_registry"], "preprocessor.pkl")
    preprocessor.save(preprocessor_path)
    
    # Save processed data for EDA dashboard usage
    processed_dir = config["paths"]["processed_data_dir"]
    os.makedirs(processed_dir, exist_ok=True)
    train_processed = X_train.copy()
    train_processed['target'] = y_train
    train_processed.to_csv(os.path.join(processed_dir, "train_processed.csv"), index=False)
    
    # Save a correlation matrix report
    reports_dir = config["paths"]["reports_dir"]
    preprocessed_reports_dir = os.path.join(reports_dir, "preprocessed")
    os.makedirs(preprocessed_reports_dir, exist_ok=True)
    corr_matrix = train_processed.corr()
    corr_matrix.to_csv(os.path.join(preprocessed_reports_dir, "correlation_matrix.csv"))
    
    # Define models dictionary
    models = {
        "Logistic_Regression": {
            "model": LogisticRegression(random_state=random_state),
            "params": config["models"]["logistic_regression"],
            "search_type": "grid"
        },
        "Decision_Tree": {
            "model": DecisionTreeClassifier(random_state=random_state),
            "params": config["models"]["decision_tree"],
            "search_type": "grid"
        },
        "Random_Forest": {
            "model": RandomForestClassifier(random_state=random_state),
            "params": config["models"]["random_forest"],
            "search_type": "random"
        },
        "XGBoost": {
            "model": xgb.XGBClassifier(random_state=random_state, use_label_encoder=False, eval_metric='logloss'),
            "params": config["models"]["xgboost"],
            "search_type": "random"
        },
        "Gradient_Boosting": {
            "model": GradientBoostingClassifier(random_state=random_state),
            "params": config["models"]["gradient_boosting"],
            "search_type": "random"
        },
        "SVM": {
            # Standard SVM is too slow for 80k rows, Calibrated LinearSVC is fast and provides predict_proba
            "model": CalibratedClassifierCV(LinearSVC(random_state=random_state, max_iter=2000)),
            "params": {}, # Skip hyperparameter grid search for SVM for execution speed
            "search_type": "none"
        },
        "KNN": {
            "model": KNeighborsClassifier(),
            "params": {}, # Skip grid search for KNN to keep execution reasonable
            "search_type": "none"
        }
    }
    
    # Set up MLflow
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])
    
    results = {}
    best_metric = -1.0
    best_model_name = None
    best_fitted_model = None
    
    # For plotting comparisons later
    plt.figure(figsize=(10, 8))
    
    # Downsample slightly for hyperparameter tuning to ensure reasonable speed on local systems
    # Using 15,000 rows is large enough for tuning and very fast.
    tune_size = min(15000, len(X_train))
    X_train_tune = X_train.iloc[:tune_size]
    y_train_tune = y_train.iloc[:tune_size]
    
    for name, m_info in models.items():
        logger.info(f"Training and tuning model: {name}...")
        
        with mlflow.start_run(run_name=name):
            clf = m_info["model"]
            params = m_info["params"]
            search_type = m_info["search_type"]
            
            if search_type == "grid" and params:
                search = GridSearchCV(clf, params, cv=3, scoring='f1', n_jobs=-1)
                search.fit(X_train_tune, y_train_tune)
                best_clf = search.best_estimator_
                mlflow.log_params(search.best_params_)
            elif search_type == "random" and params:
                search = RandomizedSearchCV(clf, params, n_iter=5, cv=3, scoring='f1', n_jobs=-1, random_state=random_state)
                search.fit(X_train_tune, y_train_tune)
                best_clf = search.best_estimator_
                mlflow.log_params(search.best_params_)
            else:
                # Direct fit on full training data (or tune_size)
                best_clf = clf
                best_clf.fit(X_train, y_train)
                
            # Final Fit on full dataset for the best estimator from search
            if search_type != "none":
                logger.info(f"Refitting best {name} model on complete training split...")
                best_clf.fit(X_train, y_train)
                
            # Predictions
            y_pred = best_clf.predict(X_test)
            y_prob = best_clf.predict_proba(X_test)[:, 1] if hasattr(best_clf, "predict_proba") else None
            
            # Metrics
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            roc_auc = roc_auc_score(y_test, y_prob) if y_prob is not None else 0.5
            
            logger.info(f"{name} Results - Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}, AUC: {roc_auc:.4f}")
            
            # Log metrics to MLflow
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", prec)
            mlflow.log_metric("recall", rec)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("roc_auc", roc_auc)
            
            # Log Model
            mlflow.sklearn.log_model(best_clf, name, serialization_format="pickle")
            
            # Store results
            results[name] = {
                "model": best_clf,
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1_score": f1,
                "roc_auc": roc_auc,
                "y_prob": y_prob
            }
            
            # Auto-selection criteria: Highest ROC-AUC (better representation of risk probability ordering)
            # Or F1 score. Let's use F1 score as primary metric since classes are highly imbalanced
            # (only ~11% are readmitted in <30 days).
            if f1 > best_metric:
                best_metric = f1
                best_model_name = name
                best_fitted_model = best_clf
                
            # Plot ROC Curve
            if y_prob is not None:
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")
                
    # Save comparison ROC Curve plot
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Comparison')
    plt.legend(loc='lower right')
    plot_dir = os.path.join(reports_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, "roc_comparison.png"), dpi=150)
    plt.close()
    
    # Save Precision-Recall curve comparison
    plt.figure(figsize=(10, 8))
    for name, r_info in results.items():
        if r_info["y_prob"] is not None:
            precision, recall, _ = precision_recall_curve(y_test, r_info["y_prob"])
            plt.plot(recall, precision, label=f"{name}")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve Comparison')
    plt.legend()
    plt.savefig(os.path.join(plot_dir, "pr_comparison.png"), dpi=150)
    plt.close()
    
    # Log best model details
    logger.info(f"BEST MODEL SELECTED: {best_model_name} with F1-Score of {best_metric:.4f}")
    
    # Save Best Model to disk
    registry_dir = config["paths"]["model_registry"]
    os.makedirs(registry_dir, exist_ok=True)
    
    model_save_path = os.path.join(registry_dir, "best_model.pkl")
    with open(model_save_path, 'wb') as f:
        pickle.dump(best_fitted_model, f)
        
    # Save a metadata JSON
    metadata = {
        "model_name": best_model_name,
        "metrics": {
            "accuracy": results[best_model_name]["accuracy"],
            "precision": results[best_model_name]["precision"],
            "recall": results[best_model_name]["recall"],
            "f1_score": results[best_model_name]["f1_score"],
            "roc_auc": results[best_model_name]["roc_auc"]
        },
        "features": preprocessor.selected_features
    }
    
    import json
    with open(os.path.join(registry_dir, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)
        
    # Generate and save confusion matrix for best model
    best_y_pred = best_fitted_model.predict(X_test)
    cm = confusion_matrix(y_test, best_y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - Best Model ({best_model_name})')
    plt.savefig(os.path.join(plot_dir, "best_confusion_matrix.png"), dpi=150)
    plt.close()
    
    # Save overall performance comparison CSV
    summary_df = pd.DataFrame(results).T.drop(columns=["model", "y_prob"])
    summary_df.to_csv(os.path.join(preprocessed_reports_dir, "model_comparison.csv"))
    
    logger.info("Pipeline Execution Completed Successfully.")
    return best_model_name, best_metric

if __name__ == "__main__":
    train_and_evaluate_pipeline()
