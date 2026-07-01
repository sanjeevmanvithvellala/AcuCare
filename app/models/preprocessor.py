import os
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from typing import Dict, List, Tuple, Any, Optional
from app.utils.logger import logger

class HealthcarePreprocessor:
    def __init__(self, target_col: str = "readmitted", random_state: int = 42):
        self.target_col = target_col
        self.random_state = random_state
        
        # Fitted attributes to be saved
        self.scaler = StandardScaler()
        self.fitted_columns = []
        self.categorical_cols = []
        self.numerical_cols = []
        self.selected_features = []
        self.label_mappings = {}
        
        # List of medication columns in the dataset
        self.medications = [
            'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
            'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
            'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
            'examide', 'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
            'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone'
        ]
        
    @staticmethod
    def map_icd9(code: Any) -> str:
        """Maps ICD-9 codes to clinical diagnostic categories."""
        if pd.isna(code) or str(code).strip() == "?" or str(code).strip() == "":
            return "Other"
            
        code_str = str(code).strip()
        
        # Check for V or E codes
        if code_str.startswith("V") or code_str.startswith("E"):
            return "Other"
            
        try:
            # Parse main numeric part of the code (ignoring decimals)
            code_num = float(code_str.split(".")[0])
            
            if 390 <= code_num <= 459 or code_num == 785:
                return "Circulatory"
            elif 460 <= code_num <= 519 or code_num == 786:
                return "Respiratory"
            elif 520 <= code_num <= 579 or code_num == 787:
                return "Digestive"
            elif int(code_num) == 250:
                return "Diabetes"
            elif 800 <= code_num <= 999:
                return "Injury"
            elif 710 <= code_num <= 739:
                return "Musculoskeletal"
            elif 580 <= code_num <= 629 or code_num == 788:
                return "Genitourinary"
            elif 140 <= code_num <= 239:
                return "Neoplasms"
            else:
                return "Other"
        except ValueError:
            return "Other"

    def clean_data(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """Cleans missing data and drops clinically irrelevant records/columns."""
        df = df.copy()
        
        # 1. Handle missing values marked as '?'
        df = df.replace('?', np.nan)
        
        # 2. Exclude patients who died or went to hospice (cannot be readmitted)
        # Discharge disposition IDs:
        # 11: Expired (died)
        # 13: Hospice / home
        # 14: Hospice / medical facility
        # 19: Expired at home
        # 20: Expired in facility
        # 21: Expired in unknown place
        # Only filter out if column exists
        if 'discharge_disposition_id' in df.columns:
            dead_or_hospice_ids = [11, 13, 14, 19, 20, 21]
            # Convert to numeric in case it's string
            df['discharge_disposition_id'] = pd.to_numeric(df['discharge_disposition_id'], errors='coerce')
            initial_count = len(df)
            df = df[~df['discharge_disposition_id'].isin(dead_or_hospice_ids)]
            if is_training:
                logger.info(f"Filtered out {initial_count - len(df)} records of deceased or hospice patients.")
                
        # 3. Drop columns with > 40% missing data
        cols_to_drop = ['weight', 'payer_code']
        # Drop if they exist
        existing_drops = [col for col in cols_to_drop if col in df.columns]
        if existing_drops:
            df = df.drop(columns=existing_drops)
            if is_training:
                logger.info(f"Dropped high-missingness columns: {existing_drops}")
                
        # 4. Impute remaining categorical missing values
        categorical_impute = ['race', 'medical_specialty']
        for col in categorical_impute:
            if col in df.columns:
                df[col] = df[col].fillna('Unknown')
                
        # 5. Handle age mapping from range string to midpoint numerical
        if 'age' in df.columns:
            age_dict = {
                '[0-10)': 5, '[10-20)': 15, '[20-30)': 25, '[30-40)': 35, '[40-50)': 45,
                '[50-60)': 55, '[60-70)': 65, '[70-80)': 75, '[80-90)': 85, '[90-100)': 95
            }
            # Fill NaNs with a default midpoint or string
            df['age_midpoint'] = df['age'].map(age_dict).fillna(55)
            # If training, log
            df = df.drop(columns=['age']) if 'age' in df.columns else df

        # 6. Map Target readmitted column
        if self.target_col in df.columns:
            # Predict 30-day readmission: <30 vs others (NO or >30)
            df['target'] = (df[self.target_col] == '<30').astype(int)
            df = df.drop(columns=[self.target_col])
            
        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies healthcare-specific feature engineering."""
        df = df.copy()
        
        # 1. Map diagnosis codes to clinical groups
        for col in ['diag_1', 'diag_2', 'diag_3']:
            if col in df.columns:
                df[f'{col}_category'] = df[col].apply(self.map_icd9)
                df = df.drop(columns=[col])
                
        # 2. Disease Severity Index
        # Combination of time_in_hospital, num_lab_procedures, and num_medications
        if all(c in df.columns for c in ['time_in_hospital', 'num_lab_procedures', 'num_medications']):
            df['disease_severity_score'] = (
                (df['time_in_hospital'] / 14.0) * 0.4 +
                (df['num_lab_procedures'] / 132.0) * 0.3 +
                (df['num_medications'] / 81.0) * 0.3
            )
            
        # 3. Patient Encounter Frequency
        # Combination of inpatient, outpatient, and emergency visits
        if all(c in df.columns for c in ['number_inpatient', 'number_outpatient', 'number_emergency']):
            df['admission_frequency'] = (
                df['number_inpatient'] * 3.0 + 
                df['number_emergency'] * 2.0 + 
                df['number_outpatient'] * 1.0
            )
            
        # 4. Medication change indicator
        # Count of how many diabetes meds were modified (Up/Down) or steady
        med_cols_present = [col for col in self.medications if col in df.columns]
        if med_cols_present:
            # Let's count changes
            df['num_medication_changes'] = df[med_cols_present].apply(
                lambda row: sum(1 for val in row if val in ['Up', 'Down']), axis=1
            )
            
        return df

    def fit(self, df: pd.DataFrame) -> 'HealthcarePreprocessor':
        """Fits the preprocessing pipeline on training data."""
        logger.info("Fitting Preprocessor...")
        
        # Clean & Engineer
        df_clean = self.clean_data(df, is_training=True)
        df_feats = self.engineer_features(df_clean)
        
        # Separate ID / Target columns
        cols_to_exclude = ['encounter_id', 'patient_nbr', 'target']
        features_df = df_feats.drop(columns=[c for c in cols_to_exclude if c in df_feats.columns])
        
        # Identify numeric and categorical columns
        self.numerical_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = features_df.select_dtypes(exclude=[np.number]).columns.tolist()
        
        logger.info(f"Numeric features found: {len(self.numerical_cols)}")
        logger.info(f"Categorical features found: {len(self.categorical_cols)}")
        
        # Fit numerical scaler
        if self.numerical_cols:
            self.scaler.fit(features_df[self.numerical_cols])
            
        # One-hot encoding preparation
        # We need to save the final column names after encoding
        encoded_sample = pd.get_dummies(features_df, columns=self.categorical_cols, drop_first=True)
        self.fitted_columns = encoded_sample.columns.tolist()
        
        # Feature Selection using ANOVA F-value (SelectKBest) if target exists
        if 'target' in df_feats.columns:
            y = df_feats['target']
            X_encoded = pd.get_dummies(features_df, columns=self.categorical_cols, drop_first=True)
            
            # Impute any NaN that might have slipped through in numeric columns
            X_encoded = X_encoded.fillna(0)
            
            # Let's select top k features (e.g. 35) or all if features are fewer
            k = min(35, X_encoded.shape[1])
            selector = SelectKBest(score_func=f_classif, k=k)
            selector.fit(X_encoded, y)
            
            # Get masks of selected features
            selected_indices = selector.get_support(indices=True)
            self.selected_features = [X_encoded.columns[i] for i in selected_indices]
            logger.info(f"Selected {len(self.selected_features)} top features via ANOVA: {self.selected_features}")
        else:
            self.selected_features = self.fitted_columns
            
        return self

    def transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Transforms a DataFrame using the fitted parameters."""
        df_clean = self.clean_data(df, is_training=False)
        df_feats = self.engineer_features(df_clean)
        
        y = df_feats['target'] if 'target' in df_feats.columns else None
        
        cols_to_exclude = ['encounter_id', 'patient_nbr', 'target']
        features_df = df_feats.drop(columns=[c for c in cols_to_exclude if c in df_feats.columns])
        
        # Ensure all fitted columns exist in the dataframe (handles partial inputs gracefully in production)
        for col in self.numerical_cols:
            if col not in features_df.columns:
                features_df[col] = 0.0
                
        for col in self.categorical_cols:
            if col not in features_df.columns:
                features_df[col] = "Unknown"

        # Impute missing values in numeric columns (simple zero/median imputation for stability)
        for col in self.numerical_cols:
            features_df[col] = features_df[col].fillna(features_df[col].median() if not features_df[col].isna().all() else 0)
                
        # Scale numeric features
        if self.numerical_cols:
            features_df[self.numerical_cols] = self.scaler.transform(features_df[self.numerical_cols])
            
        # One-hot encode categorical features
        X_encoded = pd.get_dummies(features_df, columns=self.categorical_cols, drop_first=True)
        
        # Reindex to match fitted columns (fills new missing categories with 0, drops unexpected categories)
        X_encoded = X_encoded.reindex(columns=self.fitted_columns, fill_value=0)
        
        # Filter to selected features only
        X_selected = X_encoded[self.selected_features]
        
        return X_selected, y

    def transform_single(self, patient_dict: Dict[str, Any]) -> pd.DataFrame:
        """Transforms a single patient record (dictionary format) for inference."""
        # Convert dictionary to DataFrame (must contain raw features)
        df = pd.DataFrame([patient_dict])
        
        # Ensure target is not expected
        if self.target_col in df.columns:
            df = df.drop(columns=[self.target_col])
            
        # Transform the single patient DataFrame
        X_trans, _ = self.transform(df)
        return X_trans

    def save(self, filepath: str) -> None:
        """Saves the preprocessor object as a pickle file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        logger.info(f"Preprocessor saved to {filepath}.")

    @classmethod
    def load(cls, filepath: str) -> 'HealthcarePreprocessor':
        """Loads a preprocessor object from a pickle file."""
        with open(filepath, 'rb') as f:
            preprocessor = pickle.load(f)
        logger.info(f"Preprocessor loaded from {filepath}.")
        return preprocessor
