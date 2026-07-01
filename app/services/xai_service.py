import numpy as np
import pandas as pd
import shap
from lime import lime_tabular
from typing import Dict, Any, List, Tuple, Optional
from app.utils.logger import logger

class HealthcareXAIService:
    def __init__(self, training_data: pd.DataFrame, feature_names: List[str]):
        """
        Initializes XAI Explainer service.
        training_data: Preprocessed training features DataFrame (used for background & LIME statistics)
        feature_names: List of strings naming the features
        """
        self.training_data = training_data
        self.feature_names = feature_names
        
        # Prepare LIME Explainer
        self.lime_explainer = lime_tabular.LimeTabularExplainer(
            training_data=training_data.astype(float).values,
            feature_names=feature_names,
            class_names=['Low Risk', 'High Risk'],
            mode='classification',
            random_state=42
        )
        
    def get_shap_explanation(self, model: Any, patient_features: pd.DataFrame) -> Dict[str, Any]:
        """
        Generates SHAP values for a single patient record.
        Returns a dictionary with base value, SHAP values, and feature values.
        """
        logger.info("Generating SHAP explanation...")
        try:
            # Choose appropriate SHAP explainer based on model class
            # Cast datasets to float to prevent dtype object / boolean conversion errors in SHAP C-extensions
            background_data = self.training_data.sample(n=min(100, len(self.training_data)), random_state=42).astype(float)
            patient_features_float = patient_features.astype(float)
            
            model_name = type(model).__name__
            if model_name in ["RandomForestClassifier", "XGBClassifier", "DecisionTreeClassifier", "GradientBoostingClassifier"]:
                explainer = shap.TreeExplainer(model, background_data)
            elif model_name in ["LogisticRegression"]:
                explainer = shap.LinearExplainer(model, background_data)
            else:
                # Fallback to Kernel or general Explainer
                # To get proba explanation, we can wrap prediction function
                def predict_proba_fn(x):
                    return model.predict_proba(x)[:, 1] if hasattr(model, "predict_proba") else model.predict(x)
                explainer = shap.KernelExplainer(predict_proba_fn, background_data)
                
            shap_values = explainer(patient_features_float)
            
            # Extract features info
            # shap_values could be a SHAP explanation object or array
            if hasattr(shap_values, "values"):
                # Format is [num_samples, num_features, num_classes] or [num_samples, num_features]
                s_vals = shap_values.values
                base_vals = shap_values.base_values
            else:
                # KernelExplainer outputs a list of arrays or array
                s_vals = shap_values
                base_vals = getattr(explainer, "expected_value", 0.5)
                
            # If multi-class/binary format (i.e. output has 2 classes in last dimension)
            if len(s_vals.shape) == 3:
                # Index 1 is for "High Risk" class
                s_vals = s_vals[:, :, 1]
                if isinstance(base_vals, (list, np.ndarray)) and len(base_vals) > 1:
                    base_vals = base_vals[1]
                    
            # Compress into patient specific structure
            patient_s_vals = s_vals[0]
            patient_feat_vals = patient_features.iloc[0].values
            
            explanation = []
            for name, val, s_val in zip(self.feature_names, patient_feat_vals, patient_s_vals):
                explanation.append({
                    "feature": name,
                    "value": float(val),
                    "shap_value": float(s_val)
                })
                
            # Sort by absolute SHAP value impact
            explanation.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
            
            return {
                "base_value": float(base_vals[0]) if isinstance(base_vals, (list, np.ndarray)) else float(base_vals),
                "shap_values": explanation
            }
        except Exception as e:
            logger.error(f"Error generating SHAP explanation: {e}", exc_info=True)
            return {"base_value": 0.5, "shap_values": [], "error": str(e)}

    def get_lime_explanation(self, model: Any, patient_features: pd.DataFrame) -> List[Tuple[str, float]]:
        """
        Generates LIME explanation for a single patient record.
        Returns a list of tuples containing (feature_condition, weight).
        """
        logger.info("Generating LIME explanation...")
        try:
            # We need predict_proba function
            predict_fn = model.predict_proba
            
            # LIME explanation for index 1 (High Risk)
            exp = self.lime_explainer.explain_instance(
                data_row=patient_features.iloc[0].astype(float),
                predict_fn=predict_fn,
                num_features=10
            )
            
            # exp.as_list() returns list of (feature_condition_string, weight)
            return exp.as_list()
        except Exception as e:
            logger.error(f"Error generating LIME explanation: {e}", exc_info=True)
            return [("Error generating explanation", 0.0)]
            
    def get_top_risk_factors(self, shap_explanation: Dict[str, Any], top_n: int = 5) -> List[Dict[str, Any]]:
        """Extracts top N positive features contributing to higher readmission risk."""
        shap_vals = shap_explanation.get("shap_values", [])
        # Only take features that increase risk (positive shap_value)
        risk_drivers = [item for item in shap_vals if item["shap_value"] > 0]
        return risk_drivers[:top_n]
