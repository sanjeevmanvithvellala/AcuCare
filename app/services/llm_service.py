import os
import google.generativeai as genai
from typing import Dict, Any, List, Optional
from app.utils.logger import logger
from app.utils.config_loader import load_config

class HealthcareLLMService:
    def __init__(self):
        config = load_config()
        self.system_prompt = config["llm"]["system_prompt"]
        self.model_name = config["llm"]["model_name"]
        
        # Look for API key in environment
        self.api_key = os.environ.get("GEMINI_API_KEY")
        self.is_client_active = False
        
        if self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                self.is_client_active = True
                logger.info("Gemini API Client configured successfully.")
            except Exception as e:
                logger.error(f"Error configuring Gemini API: {e}")
        else:
            logger.info("GEMINI_API_KEY not found. Using Clinical Expert Rule-Based Fallback Engine.")

    def generate_clinical_explanation(
        self, 
        patient_raw: Dict[str, Any], 
        probability: float, 
        prediction: int,
        top_features: List[Dict[str, Any]]
    ) -> str:
        """Generates a plain-English clinical explanation of the readmission risk."""
        risk_level = "High" if probability >= 0.5 else "Medium" if probability >= 0.2 else "Low"
        
        if self.is_client_active:
            try:
                prompt = self._build_explanation_prompt(patient_raw, probability, risk_level, top_features)
                
                # Try generating content using config model
                model = genai.GenerativeModel(
                    model_name=self.model_name,
                    system_instruction=self.system_prompt
                )
                response = model.generate_content(prompt)
                return response.text
            except Exception as e:
                logger.warning(f"Gemini API request failed ({e}). Falling back to rule-based generation.")
                # Fallback to rule-based engine on error
                
        return self._generate_rule_based_explanation(patient_raw, probability, risk_level, top_features)

    def chat_about_patient(
        self,
        patient_raw: Dict[str, Any],
        probability: float,
        top_features: List[Dict[str, Any]],
        chat_history: List[Dict[str, str]],
        user_message: str
    ) -> str:
        """Allows clinical users to ask questions about a patient's risk profile."""
        risk_level = "High" if probability >= 0.5 else "Medium" if probability >= 0.2 else "Low"
        
        if self.is_client_active:
            try:
                # Prepare a conversation context prompt
                context_prompt = self._build_chat_context(patient_raw, probability, risk_level, top_features)
                
                # Format history for Gemini API
                contents = []
                # Inject initial context as part of the system or first message
                contents.append({"role": "user", "parts": [f"Patient context:\n{context_prompt}"]})
                contents.append({"role": "model", "parts": ["Understood. I have reviewed the patient's risk profile and am ready to answer any questions you have regarding their 30-day readmission risk."]})
                
                # Append subsequent chat history
                for chat in chat_history:
                    role = "user" if chat["role"] == "user" else "model"
                    contents.append({"role": role, "parts": [chat["content"]]})
                    
                # Append user's current message
                contents.append({"role": "user", "parts": [user_message]})
                
                # Run model
                model = genai.GenerativeModel(
                    model_name=self.model_name,
                    system_instruction="You are a clinical AI consultant. Answer questions about the patient's risk prediction accurately based on the patient data, SHAP features, and medical guidelines."
                )
                response = model.generate_content(contents)
                return response.text
            except Exception as e:
                logger.error(f"Chat failed with Gemini API: {e}")
                return f"Gemini API Error. Fallback response: I am unable to connect to the LLM backend at the moment. However, reviewing the rules, the patient is classified as {risk_level} Risk ({probability*100:.1f}% probability) largely due to {', '.join([f['feature'] for f in top_features[:3]])}."
                
        return self._generate_rule_based_chat_response(patient_raw, probability, risk_level, top_features, user_message)

    def _build_explanation_prompt(
        self, 
        patient_raw: Dict[str, Any], 
        probability: float, 
        risk_level: str, 
        top_features: List[Dict[str, Any]]
    ) -> str:
        """Helper to structure the prompt for generating explanations."""
        features_summary = "\n".join([
            f"- {item['feature']}: value = {item['value']:.2f} (SHAP impact = +{item['shap_value']:.4f})"
            for item in top_features
        ])
        
        # Clean patient dict of keys with NaN or None for cleaner prompt
        clean_patient = {k: v for k, v in patient_raw.items() if v is not None and not (isinstance(v, float) and np.isnan(v))}
        
        prompt = f"""
        Patient Demographics & Clinical Metrics:
        {clean_patient}
        
        Prediction Summary:
        - 30-day readmission risk: {risk_level}
        - Risk Probability: {probability * 100:.1f}%
        
        Top Risk Drivers (SHAP values):
        {features_summary}
        
        Please generate the clinical explanation report.
        """
        return prompt

    def _build_chat_context(
        self,
        patient_raw: Dict[str, Any],
        probability: float,
        risk_level: str,
        top_features: List[Dict[str, Any]]
    ) -> str:
        """Helper to structure patient context for chatbot."""
        clean_patient = {k: v for k, v in patient_raw.items() if v is not None and not (isinstance(v, float) and np.isnan(v))}
        features_summary = ", ".join([f"{item['feature']} (+{item['shap_value']:.2f})" for item in top_features[:5]])
        
        context = (
            f"The patient is a {clean_patient.get('age', 'Unknown age')} {clean_patient.get('race', 'Unknown race')} "
            f"{clean_patient.get('gender', 'patient')}. They were hospitalized for {clean_patient.get('time_in_hospital', 'N/A')} days. "
            f"Their predicted 30-day hospital readmission risk is {risk_level} ({probability*100:.1f}% probability). "
            f"The top clinical features contributing to this risk are: {features_summary}."
        )
        return context

    def _generate_rule_based_explanation(
        self,
        patient_raw: Dict[str, Any],
        probability: float,
        risk_level: str,
        top_features: List[Dict[str, Any]]
    ) -> str:
        """Rule-based clinical explanation generator (fallback)."""
        # Determine main drivers
        drivers = []
        for feat in top_features[:3]:
            name = feat["feature"]
            val = feat["value"]
            if "number_inpatient" in name:
                drivers.append(f"a high number of inpatient admissions in the preceding year ({int(val)} stays)")
            elif "disease_severity" in name:
                drivers.append("a high clinical disease severity score (based on hospital stay, medications, and labs)")
            elif "number_emergency" in name:
                drivers.append(f"frequent emergency room visits ({int(val)} visits)")
            elif "num_medication_changes" in name:
                drivers.append(f"multiple changes in diabetes medications during the hospital stay ({int(val)} modifications)")
            elif "number_diagnoses" in name:
                drivers.append(f"multiple active comorbidities ({int(val)} diagnoses listed)")
            elif "time_in_hospital" in name:
                drivers.append(f"an extended index hospital stay ({int(val)} days)")
            else:
                drivers.append(f"elevated value for '{name.replace('_', ' ')}' ({val:.2f})")
                
        driver_str = ", ".join(drivers)
        if len(drivers) > 1:
            # Replace last comma with "and"
            r_idx = driver_str.rfind(", ")
            if r_idx != -1:
                driver_str = driver_str[:r_idx] + ", and " + driver_str[r_idx+2:]
                
        explanation = (
            f"### Clinical Explanation Report\n\n"
            f"**Risk Level Summary:**\n"
            f"The model predicts this patient has a **{risk_level} Risk** of readmission within 30 days of discharge, "
            f"with a probability of **{probability*100:.1f}%**.\n\n"
            f"**Key Clinical Drivers:**\n"
            f"This prediction is primarily driven by: **{driver_str}**.\n"
            f"Statistical analysis of prior hospital discharges indicates that patients with these specific indicators "
            f"experience significantly higher post-discharge vulnerability, requiring structured transition-of-care support.\n\n"
            f"**Actionable Clinical Recommendations:**\n"
        )
        
        # Tailored recommendations based on features
        recommendations = []
        has_inpatient = any("inpatient" in f["feature"] for f in top_features)
        has_changes = any("medication_changes" in f["feature"] or "med_" in f["feature"] for f in top_features)
        has_severity = any("severity" in f["feature"] for f in top_features)
        
        if has_inpatient:
            recommendations.append("- **Transition of Care Coordinator**: Assign a dedicated transitional care manager to follow this patient. Schedule a home visit within 72 hours of discharge.")
            recommendations.append("- **Post-Discharge Contact**: Conduct telephone follow-up calls at 24 hours, 7 days, and 14 days post-discharge to verify care plans.")
        if has_changes:
            recommendations.append("- **Pharmacist Medication Reconciliation**: Execute a comprehensive medication review at discharge. Provide structured counseling on the modified diabetes regimen.")
            recommendations.append("- **Glucose Monitoring Protocol**: Set up daily home blood glucose monitoring logs, with immediate escalation criteria for values outside the target range.")
        if has_severity:
            recommendations.append("- **Multidisciplinary Consults**: Ensure outpatient consults with Endocrinology and Cardiology are scheduled within 7 to 10 days.")
            recommendations.append("- **Patient & Family Education**: Deliver patient-centered education regarding red-flag symptoms (e.g. polyuria, polydipsia, severe dizziness) requiring urgent clinic contact.")
            
        if not recommendations:
            recommendations.append("- **Early Outpatient Follow-up**: Arrange a primary care provider follow-up appointment within 7 days of hospital discharge.")
            recommendations.append("- **Red-Flag Symptoms**: Educate the patient on early signs of complications and provide contact numbers for 24/7 clinical support.")
            
        explanation += "\n".join(recommendations)
        return explanation

    def _generate_rule_based_chat_response(
        self,
        patient_raw: Dict[str, Any],
        probability: float,
        risk_level: str,
        top_features: List[Dict[str, Any]],
        user_message: str
    ) -> str:
        """Generates simple rule-based responses to standard clinical queries (fallback)."""
        msg = user_message.lower()
        
        if "why" in msg or "risk" in msg or "driver" in msg:
            drivers = [f"{item['feature']} (value: {item['value']:.2f})" for item in top_features[:3]]
            return (
                f"The patient is classified as **{risk_level} Risk** ({probability*100:.1f}% probability) "
                f"primarily due to the following top risk factors:\n" + 
                "\n".join([f"- **{d}**" for d in drivers]) + 
                "\n\nThese features increase the model's output probability above the baseline rate."
            )
        elif "recommend" in msg or "action" in msg or "prevent" in msg or "help" in msg:
            return (
                "Based on the patient's risk profile, the following clinical interventions are recommended:\n"
                "1. **Early Primary Care Appointment**: Schedule a follow-up visit within 7 days of discharge.\n"
                "2. **Medication Reconciliation**: Have a clinical pharmacist review all active prescriptions and counseling on diabetes medication changes.\n"
                "3. **Transitional Care Management**: Assign a nurse or case coordinator for post-discharge phone calls at 48 hours."
            )
        elif "age" in msg or "demographic" in msg:
            return (
                f"The patient's age is recorded in the midpoint range of **{patient_raw.get('age_midpoint', 'N/A')} years**. "
                f"Their race is **{patient_raw.get('race', 'Unknown')}** and gender is **{patient_raw.get('gender', 'N/A')}**."
            )
        else:
            return (
                "I am running in Clinical Expert Fallback mode. I can answer questions about the patient's "
                "risk factors, demographics, and clinical recommendations. Could you clarify your question?"
            )
