from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__)
CORS(app)

# Global variables
model = None
scaler = None
label_encoders = None
feature_columns = None


def load_model():
    global model, scaler, label_encoders, feature_columns
    try:
        # Try to find model in different locations
        model_paths = [
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "healthcare_ml_api",
                "models",
                "readmission_model.pkl",
            ),
            os.path.join(
                os.path.dirname(__file__), "..", "models", "readmission_model.pkl"
            ),
            "healthcare_ml_api/models/readmission_model.pkl",
            "models/readmission_model.pkl",
        ]

        model_path = None
        for path in model_paths:
            if os.path.exists(path):
                model_path = path
                break

        if model_path is None:
            print("Model file not found")
            return False

        model_artifacts = joblib.load(model_path)
        model = model_artifacts["model"]
        scaler = model_artifacts["scaler"]
        label_encoders = model_artifacts["label_encoders"]
        feature_columns = model_artifacts["feature_columns"]
        print(f"Model loaded successfully from {model_path}")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False


# Load model on startup
load_model()


def preprocess_input(data):
    try:
        df = pd.DataFrame([data])

        # Ensure all required features are present
        if feature_columns:
            for col in feature_columns:
                if col not in df.columns:
                    df[col] = 0

        # Apply label encoding
        if label_encoders:
            for col, encoder in label_encoders.items():
                if col in df.columns:
                    try:
                        df[col] = encoder.transform(df[col])
                    except ValueError:
                        df[col] = -1

        # Scale numerical features
        if scaler and hasattr(scaler, "feature_names_in_"):
            numerical_features = scaler.feature_names_in_.tolist()
            numerical_features = [f for f in numerical_features if f in df.columns]
            if numerical_features:
                df[numerical_features] = scaler.transform(df[numerical_features])

        if feature_columns:
            return df[feature_columns], None
        return df, None
    except Exception as e:
        return None, str(e)


def generate_recommendations(risk_probability, patient_data):
    recommendations = []

    if risk_probability > 0.7:
        recommendations.extend(
            [
                "Schedule follow-up appointment within 7 days",
                "Review medication adherence and adjust if needed",
                "Assign care coordinator for discharge planning",
                "Consider home health visits",
                "Ensure clear discharge instructions with caregiver",
            ]
        )
    elif risk_probability > 0.3:
        recommendations.extend(
            [
                "Schedule follow-up appointment within 14 days",
                "Provide written discharge instructions",
                "Review medications before discharge",
                "Consider telehealth check-in at day 7",
            ]
        )
    else:
        recommendations.extend(
            ["Standard discharge planning", "Provide patient education materials"]
        )

    # Additional specific recommendations
    if patient_data.get("age", 0) > 75:
        recommendations.append("Geriatric consultation recommended")

    if patient_data.get("comorbidity_count", 0) > 3:
        recommendations.append("Review complex medication regimen")

    return recommendations


@app.route("/", methods=["POST", "OPTIONS"])
def predict():
    if model is None:
        return (
            jsonify({"error": "Model not loaded. Please ensure model file exists."}),
            503,
        )

    if request.method == "OPTIONS":
        return jsonify({}), 200

    try:
        data = request.json

        # Validate input
        required_fields = [
            "age",
            "length_of_stay_days",
            "comorbidity_count",
            "admission_type",
        ]
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({"error": f"Missing required fields: {missing_fields}"}), 400

        # Preprocess input
        processed_data, error = preprocess_input(data)
        if error:
            return jsonify({"error": error}), 400

        # Make prediction
        prediction = model.predict(processed_data)[0]
        probability = model.predict_proba(processed_data)[0][1]

        # Prepare response
        response = {
            "prediction": int(prediction),
            "readmission_risk": float(probability),
            "risk_level": (
                "High"
                if probability > 0.7
                else "Medium" if probability > 0.3 else "Low"
            ),
            "confidence_interval": {
                "lower": max(0, probability - 0.1),
                "upper": min(1, probability + 0.1),
            },
            "recommendations": generate_recommendations(probability, data),
            "timestamp": datetime.now().isoformat(),
            "model_version": "v1.0.0",
        }

        return jsonify(response)

    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return jsonify({"error": str(e)}), 500


# Vercel handler
handler = app
