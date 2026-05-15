# monitoring.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import deque
import json
import threading
import time

class ModelMonitor:
    def __init__(self, max_history=10000):
        self.predictions = deque(maxlen=max_history)
        self.actual_outcomes = {}
        self.drift_detection_window = 1000
        self.alert_threshold = 0.05
        self.lock = threading.Lock()
        
    def log_prediction(self, input_data, prediction, probability):
        """Log prediction for monitoring"""
        with self.lock:
            prediction_record = {
                'timestamp': datetime.now().isoformat(),
                'prediction': int(prediction),
                'probability': float(probability),
                'risk_level': 'High' if probability > 0.7 else 'Medium' if probability > 0.3 else 'Low',
                'input_summary': self._summarize_input(input_data),
                'prediction_id': len(self.predictions)
            }
            self.predictions.append(prediction_record)
            
            # Check for drift
            self._check_prediction_drift()
            
            # Log to file
            self._write_to_log(prediction_record)
    
    def log_actual_outcome(self, prediction_id, actual_readmission):
        """Log actual outcome for model validation"""
        with self.lock:
            self.actual_outcomes[prediction_id] = {
                'actual': actual_readmission,
                'timestamp': datetime.now().isoformat()
            }
    
    def _summarize_input(self, input_data):
        """Create a summary of input data for monitoring"""
        summary = {}
        risk_factors = ['age', 'comorbidity_count', 'previous_admissions_90d', 'length_of_stay_days']
        for factor in risk_factors:
            if factor in input_data:
                summary[factor] = input_data[factor]
        return summary
    
    def _write_to_log(self, record):
        """Write prediction to log file"""
        try:
            with open('logs/predictions.log', 'a') as f:
                f.write(json.dumps(record) + '\n')
        except Exception as e:
            print(f"Error writing to log: {e}")
    
    def _check_prediction_drift(self):
        """Check for prediction drift"""
        if len(self.predictions) < self.drift_detection_window:
            return
        
        recent = list(self.predictions)[-self.drift_detection_window:]
        historical = list(self.predictions)[:-self.drift_detection_window]
        
        recent_rate = sum(p['prediction'] for p in recent) / len(recent)
        historical_rate = sum(p['prediction'] for p in historical) / len(historical)
        
        if abs(recent_rate - historical_rate) > self.alert_threshold:
            self._send_alert(f"Prediction drift detected: Recent rate {recent_rate:.2%} vs Historical {historical_rate:.2%}")
    
    def _send_alert(self, message):
        """Send monitoring alert"""
        alert = {
            'timestamp': datetime.now().isoformat(),
            'type': 'drift_alert',
            'message': message,
            'severity': 'WARNING'
        }
        
        print(f"ALERT: {message}")
        
        # Write alert to file
        try:
            with open('logs/alerts.log', 'a') as f:
                f.write(json.dumps(alert) + '\n')
        except Exception as e:
            print(f"Error writing alert: {e}")
    
    def get_recent_predictions(self, limit=10):
        """Get recent predictions for dashboard display"""
        with self.lock:
            recent = list(self.predictions)[-limit:]
            # Reverse to show newest first
            recent.reverse()
            return [{
                'timestamp': p['timestamp'],
                'prediction': p['prediction'],
                'probability': p['probability'],
                'risk_level': p['risk_level'],
                'age': p['input_summary'].get('age', 'N/A')
            } for p in recent]
    
    def get_metrics(self):
        """Get current monitoring metrics"""
        with self.lock:
            metrics = {
                'total_predictions': len(self.predictions),
                'recent_predictions': len(list(self.predictions)[-1000:]),
                'prediction_distribution': self._get_prediction_distribution(),
                'risk_level_distribution': self._get_risk_distribution(),
                'performance_metrics': self._calculate_performance_metrics(),
                'data_drift_indicators': self._calculate_data_drift(),
                'timestamp': datetime.now().isoformat()
            }
            return metrics
    
    def _get_prediction_distribution(self):
        """Get distribution of predictions"""
        if not self.predictions:
            return {'no_readmission': 0, 'readmission': 0, 'readmission_rate': 0}
        
        predictions = [p['prediction'] for p in self.predictions]
        readmissions = predictions.count(1)
        return {
            'no_readmission': len(predictions) - readmissions,
            'readmission': readmissions,
            'readmission_rate': readmissions / len(predictions) if len(predictions) > 0 else 0
        }
    
    def _get_risk_distribution(self):
        """Get distribution of risk levels"""
        if not self.predictions:
            return {'high_risk': 0, 'medium_risk': 0, 'low_risk': 0}
        
        risks = []
        for p in self.predictions:
            prob = p['probability']
            if prob > 0.7:
                risks.append('high')
            elif prob > 0.3:
                risks.append('medium')
            else:
                risks.append('low')
        
        return {
            'high_risk': risks.count('high'),
            'medium_risk': risks.count('medium'),
            'low_risk': risks.count('low')
        }
    
    def _calculate_performance_metrics(self):
        """Calculate performance metrics if actual outcomes available"""
        if not self.actual_outcomes:
            return {'status': 'Insufficient data for performance calculation', 'accuracy': None}
        
        # Calculate accuracy, precision, recall
        correct = 0
        tp = fp = tn = fn = 0
        
        for pred_id, outcome in self.actual_outcomes.items():
            if pred_id < len(self.predictions):
                prediction = self.predictions[pred_id]['prediction']
                actual = outcome['actual']
                
                if prediction == actual:
                    correct += 1
                
                if prediction == 1 and actual == 1:
                    tp += 1
                elif prediction == 1 and actual == 0:
                    fp += 1
                elif prediction == 0 and actual == 0:
                    tn += 1
                elif prediction == 0 and actual == 1:
                    fn += 1
        
        total = len(self.actual_outcomes)
        accuracy = correct / total if total > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'total_validated': total,
            'confusion_matrix': {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}
        }
    
    def _calculate_data_drift(self):
        """Calculate data drift indicators"""
        if len(self.predictions) < 100:
            return {'status': 'Need more data for drift detection'}
        
        recent = list(self.predictions)[-100:]
        historical = list(self.predictions)[-1000:-100] if len(self.predictions) >= 1000 else []
        
        if not historical:
            return {'status': 'Need more historical data'}
        
        drift_indicators = {}
        
        # Check age distribution drift
        recent_ages = [p['input_summary'].get('age', 0) for p in recent if 'age' in p['input_summary']]
        historical_ages = [p['input_summary'].get('age', 0) for p in historical if 'age' in p['input_summary']]
        
        if recent_ages and historical_ages:
            drift_indicators['age_drift'] = abs(np.mean(recent_ages) - np.mean(historical_ages))
        
        return drift_indicators