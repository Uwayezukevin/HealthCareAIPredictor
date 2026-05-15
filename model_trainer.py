# model_trainer.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import joblib
import warnings
warnings.filterwarnings('ignore')

class ReadmissionModelTrainer:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.feature_columns = None
        
    def generate_synthetic_data(self, n_samples=10000):
        """Generate synthetic patient data for demonstration"""
        np.random.seed(42)
        
        data = {
            'age': np.random.randint(18, 95, n_samples),
            'gender': np.random.choice(['M', 'F'], n_samples),
            'length_of_stay_days': np.random.exponential(5, n_samples),
            'comorbidity_count': np.random.poisson(2, n_samples),
            'previous_admissions_90d': np.random.poisson(0.5, n_samples),
            'medications_count': np.random.poisson(3, n_samples),
            'hemoglobin_level': np.random.normal(13, 2, n_samples),
            'white_blood_cell_count': np.random.normal(7.5, 3, n_samples),
            'creatinine_level': np.random.normal(0.9, 0.3, n_samples),
            'systolic_bp': np.random.normal(120, 15, n_samples),
            'heart_rate': np.random.normal(75, 12, n_samples),
            'emergency_visits_last_30d': np.random.poisson(0.3, n_samples),
            'admission_type': np.random.choice(['Emergency', 'Urgent', 'Elective'], n_samples),
            'insurance_type': np.random.choice(['Private', 'Medicare', 'Medicaid', 'None'], n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Generate target (hospital readmission) based on risk factors
        risk_score = (
            (df['previous_admissions_90d'] > 1) * 0.3 +
            (df['comorbidity_count'] > 3) * 0.2 +
            (df['emergency_visits_last_30d'] > 0) * 0.2 +
            (df['hemoglobin_level'] < 10) * 0.15 +
            (df['age'] > 75) * 0.1 +
            (df['length_of_stay_days'] > 7) * 0.05
        )
        
        df['readmission_30day'] = (np.random.random(n_samples) < risk_score).astype(int)
        
        return df
    
    def create_preprocessing_pipeline(self, categorical_features, numerical_features):
        """Create preprocessing pipeline for features"""
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('label_encoder', 'passthrough')  # Will handle with custom encoder
        ])
        
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', 'passthrough', categorical_features)
            ])
        
        return preprocessor
    
    def train_model(self, df=None):
        """Train the machine learning model"""
        
        if df is None:
            df = self.generate_synthetic_data()
        
        # Define features
        self.feature_columns = [col for col in df.columns if col != 'readmission_30day']
        
        # Separate features and target
        X = df[self.feature_columns]
        y = df['readmission_30day']
        
        # Identify categorical and numerical features
        categorical_features = X.select_dtypes(include=['object']).columns.tolist()
        numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
        
        # Create label encoders for categorical features
        self.label_encoders = {}
        for cat_feat in categorical_features:
            self.label_encoders[cat_feat] = LabelEncoder()
            X[cat_feat] = self.label_encoders[cat_feat].fit_transform(X[cat_feat])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale numerical features
        self.scaler = StandardScaler()
        X_train[numerical_features] = self.scaler.fit_transform(X_train[numerical_features])
        X_test[numerical_features] = self.scaler.transform(X_test[numerical_features])
        
        # Train Random Forest Classifier
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        print("=== Model Performance ===")
        print(f"AUC-ROC Score: {roc_auc_score(y_test, y_pred_proba):.3f}")
        print(f"Cross-validation Score: {cross_val_score(self.model, X_train, y_train, cv=5).mean():.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Important Features:")
        print(feature_importance.head(10))
        
        # Save model and preprocessing objects
        self.save_model()
        
        return self.model
    
    def save_model(self):
        """Save model and preprocessing objects"""
        model_artifacts = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns,
            'numerical_features': self.scaler.feature_names_in_.tolist() if hasattr(self.scaler, 'feature_names_in_') else None
        }
        joblib.dump(model_artifacts, 'models/readmission_model.pkl')
        print("Model saved to 'models/readmission_model.pkl'")

if __name__ == "__main__":
    trainer = ReadmissionModelTrainer()
    trainer.train_model()