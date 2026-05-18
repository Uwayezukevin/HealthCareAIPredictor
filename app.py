# app.py - Complete working version for Render
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from datetime import datetime
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# ============================================
# SIMPLE MONITORING CLASS (No external file needed)
# ============================================
class SimpleMonitor:
    def __init__(self):
        self.predictions = []
    
    def log_prediction(self, data, prediction, probability):
        """Log a prediction for monitoring"""
        self.predictions.append({
            'timestamp': datetime.now().isoformat(),
            'data': data,
            'prediction': prediction,
            'probability': probability
        })
        # Keep only last 100 predictions
        if len(self.predictions) > 100:
            self.predictions.pop(0)
    
    def get_metrics(self):
        """Get monitoring metrics"""
        if not self.predictions:
            return {
                'total_predictions': 0,
                'prediction_distribution': {'readmission_rate': 0},
                'risk_level_distribution': {'high_risk': 0, 'medium_risk': 0, 'low_risk': 0},
                'performance_metrics': {'accuracy': None, 'status': 'Monitoring active'}
            }
        
        total = len(self.predictions)
        readmissions = sum(1 for p in self.predictions if p['prediction'] == 1)
        
        # Calculate risk levels based on probability
        high_risk = sum(1 for p in self.predictions if p['probability'] > 0.7)
        medium_risk = sum(1 for p in self.predictions if 0.3 < p['probability'] <= 0.7)
        low_risk = sum(1 for p in self.predictions if p['probability'] <= 0.3)
        
        return {
            'total_predictions': total,
            'prediction_distribution': {
                'readmission_rate': readmissions / total if total > 0 else 0
            },
            'risk_level_distribution': {
                'high_risk': high_risk,
                'medium_risk': medium_risk,
                'low_risk': low_risk
            },
            'performance_metrics': {'accuracy': None, 'status': 'Collecting data'}
        }
    
    def get_recent_predictions(self, limit=10):
        """Get recent predictions for dashboard"""
        recent = list(self.predictions)[-limit:]
        recent.reverse()  # Show newest first
        return [{
            'timestamp': p['timestamp'],
            'prediction': p['prediction'],
            'probability': p['probability'],
            'risk_level': 'High' if p['probability'] > 0.7 else 'Medium' if p['probability'] > 0.3 else 'Low',
            'age': p['data'].get('age', 'N/A')
        } for p in recent]

# Initialize monitor
monitor = SimpleMonitor()

# ============================================
# RECOMMENDATION ENGINE
# ============================================
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

# ============================================
# RISK CALCULATION ENGINE (Rule-based)
# ============================================
def calculate_risk_score(patient_data):
    """Calculate risk score based on patient data"""
    risk_score = 0.0
    
    # Age factor
    age = patient_data.get('age', 0)
    if age > 75:
        risk_score += 0.3
    elif age > 65:
        risk_score += 0.2
    elif age > 50:
        risk_score += 0.1
    
    # Length of stay factor
    los = patient_data.get('length_of_stay_days', 0)
    if los > 10:
        risk_score += 0.3
    elif los > 5:
        risk_score += 0.2
    elif los > 3:
        risk_score += 0.1
    
    # Comorbidity factor
    comorbidities = patient_data.get('comorbidity_count', 0)
    if comorbidities > 4:
        risk_score += 0.3
    elif comorbidities > 2:
        risk_score += 0.2
    elif comorbidities > 0:
        risk_score += 0.1
    
    # Admission type factor
    admission_type = patient_data.get('admission_type', '')
    if admission_type == 'Emergency':
        risk_score += 0.2
    elif admission_type == 'Urgent':
        risk_score += 0.1
    
    # Previous admissions factor
    prev_admissions = patient_data.get('previous_admissions_90d', 0)
    if prev_admissions > 2:
        risk_score += 0.2
    elif prev_admissions > 0:
        risk_score += 0.1
    
    # Cap the risk score at 0.95
    return min(risk_score, 0.95)

# ============================================
# API ENDPOINTS
# ============================================
@app.route('/')
def dashboard():
    """Main dashboard page"""
    try:
        return render_template('dashboard.html', metrics=monitor.get_metrics())
    except Exception as e:
        logger.warning(f"Template not found: {e}")
        return jsonify({
            'message': 'Healthcare AI Predictor API',
            'status': 'running',
            'endpoints': ['/health', '/predict', '/batch_predict', '/model/metrics', '/model/recent_predictions']
        })

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': True,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Single patient prediction endpoint"""
    try:
        data = request.json
        
        # Validate required fields
        required_fields = ['age', 'length_of_stay_days', 'comorbidity_count', 'admission_type']
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({
                'error': f'Missing required fields: {missing_fields}'
            }), 400
        
        # Calculate risk
        probability = calculate_risk_score(data)
        prediction = 1 if probability > 0.5 else 0
        
        # Log for monitoring
        monitor.log_prediction(data, prediction, probability)
        
        # Prepare response
        response = {
            'prediction': prediction,
            'readmission_risk': probability,
            'risk_level': 'High' if probability > 0.7 else 'Medium' if probability > 0.3 else 'Low',
            'confidence_interval': {
                'lower': max(0, probability - 0.1),
                'upper': min(1, probability + 0.1)
            },
            'recommendations': generate_recommendations(probability, data),
            'timestamp': datetime.now().isoformat(),
            'model_version': 'v1.0.0'
        }
        
        logger.info(f"Prediction: {response['risk_level']} risk for age {data.get('age')}")
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Batch prediction endpoint"""
    try:
        data = request.json
        
        if 'patients' not in data:
            return jsonify({'error': 'Missing patients array'}), 400
        
        patients = data['patients']
        results = []
        
        for idx, patient in enumerate(patients):
            try:
                probability = calculate_risk_score(patient)
                prediction = 1 if probability > 0.5 else 0
                
                results.append({
                    'index': idx,
                    'prediction': prediction,
                    'readmission_risk': probability,
                    'risk_level': 'High' if probability > 0.7 else 'Medium' if probability > 0.3 else 'Low'
                })
                
                # Log each prediction
                monitor.log_prediction(patient, prediction, probability)
                
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

@app.route('/model/metrics')
def get_model_metrics():
    """Get monitoring metrics"""
    return jsonify(monitor.get_metrics())

@app.route('/model/recent_predictions')
def get_recent_predictions():
    """Get recent predictions"""
    limit = request.args.get('limit', 10, type=int)
    return jsonify(monitor.get_recent_predictions(limit))

# ============================================
# MAIN ENTRY POINT
# ============================================
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)