# app.py
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import logging
from monitoring import ModelMonitor
import traceback

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Initialize monitor
monitor = ModelMonitor()

# Load model and artifacts
try:
    model_artifacts = joblib.load('models/readmission_model.pkl')
    model = model_artifacts['model']
    scaler = model_artifacts['scaler']
    label_encoders = model_artifacts['label_encoders']
    feature_columns = model_artifacts['feature_columns']
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model = None

def preprocess_input(data):
    """Preprocess incoming data for prediction"""
    try:
        # Convert to DataFrame
        df = pd.DataFrame([data])
        
        # Ensure all required features are present
        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0  # Default value for missing features
        
        # Apply label encoding to categorical features
        for col, encoder in label_encoders.items():
            if col in df.columns:
                try:
                    df[col] = encoder.transform(df[col])
                except ValueError:
                    # Handle unknown categories
                    df[col] = -1
        
        # Separate numerical and categorical
        numerical_features = scaler.feature_names_in_.tolist() if hasattr(scaler, 'feature_names_in_') else []
        numerical_features = [f for f in numerical_features if f in df.columns]
        
        if numerical_features:
            df[numerical_features] = scaler.transform(df[numerical_features])
        
        return df[feature_columns], None
    
    except Exception as e:
        return None, str(e)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint for single patient prediction"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 503
    
    try:
        data = request.json
        
        # Validate input
        required_fields = ['age', 'length_of_stay_days', 'comorbidity_count', 'admission_type']
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({
                'error': f'Missing required fields: {missing_fields}'
            }), 400
        
        # Preprocess input
        processed_data, error = preprocess_input(data)
        if error:
            return jsonify({'error': error}), 400
        
        # Make prediction
        prediction = model.predict(processed_data)[0]
        probability = model.predict_proba(processed_data)[0][1]
        
        # Log prediction for monitoring
        monitor.log_prediction(data, prediction, probability)
        
        # Prepare response
        response = {
            'prediction': int(prediction),
            'readmission_risk': float(probability),
            'risk_level': 'High' if probability > 0.7 else 'Medium' if probability > 0.3 else 'Low',
            'confidence_interval': {
                'lower': max(0, probability - 0.1),
                'upper': min(1, probability + 0.1)
            },
            'recommendations': generate_recommendations(probability, data),
            'timestamp': datetime.now().isoformat(),
            'model_version': 'v1.0.0'
        }
        
        logger.info(f"Prediction made for patient: {response['risk_level']} risk")
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Prediction error: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Endpoint for batch predictions"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 503
    
    try:
        data = request.json
        
        if 'patients' not in data:
            return jsonify({'error': 'Missing patients array'}), 400
        
        patients = data['patients']
        results = []
        
        for idx, patient in enumerate(patients):
            try:
                processed_data, error = preprocess_input(patient)
                if error:
                    results.append({'index': idx, 'error': error})
                    continue
                
                prediction = model.predict(processed_data)[0]
                probability = model.predict_proba(processed_data)[0][1]
                
                results.append({
                    'index': idx,
                    'prediction': int(prediction),
                    'readmission_risk': float(probability),
                    'risk_level': 'High' if probability > 0.7 else 'Medium' if probability > 0.3 else 'Low'
                })
                
            except Exception as e:
                results.append({'index': idx, 'error': str(e)})
        
        return jsonify({
            'batch_id': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'total_patients': len(patients),
            'successful_predictions': len([r for r in results if 'error' not in r]),
            'results': results,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def generate_recommendations(risk_probability, patient_data):
    """Generate care recommendations based on risk level"""
    recommendations = []
    
    if risk_probability > 0.7:
        recommendations.extend([
            "Schedule follow-up appointment within 7 days",
            "Review medication adherence and adjust if needed",
            "Assign care coordinator for discharge planning",
            "Consider home health visits",
            "Ensure clear discharge instructions with caregiver"
        ])
    elif risk_probability > 0.3:
        recommendations.extend([
            "Schedule follow-up appointment within 14 days",
            "Provide written discharge instructions",
            "Review medications before discharge",
            "Consider telehealth check-in at day 7"
        ])
    else:
        recommendations.extend([
            "Standard discharge planning",
            "Provide patient education materials"
        ])
    
    # Additional specific recommendations
    if patient_data.get('age', 0) > 75:
        recommendations.append("Geriatric consultation recommended")
    
    if patient_data.get('comorbidity_count', 0) > 3:
        recommendations.append("Review complex medication regimen")
    
    return recommendations
@app.route('/model/recent_predictions', methods=['GET'])
def get_recent_predictions():
    """Get recent predictions for dashboard"""
    limit = request.args.get('limit', 10, type=int)
    return jsonify(monitor.get_recent_predictions(limit))

@app.route('/model/metrics', methods=['GET'])
def get_model_metrics():
    """Get model performance metrics"""
    return jsonify(monitor.get_metrics())

@app.route('/', methods=['GET'])
def dashboard():
    """Simple monitoring dashboard"""
    return render_template('dashboard.html', metrics=monitor.get_metrics())

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)