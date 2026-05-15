# test_api.py
import requests
import json
import time

# Test health endpoint
def test_health():
    print("\n=== Testing Health Check ===")
    try:
        response = requests.get("http://localhost:5000/health")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

# Test single prediction
def test_single_prediction():
    print("\n=== Testing Single Prediction ===")
    url = "http://localhost:5000/predict"
    
    patient_data = {
        "age": 72,
        "gender": "F",
        "length_of_stay_days": 8,
        "comorbidity_count": 4,
        "previous_admissions_90d": 2,
        "medications_count": 5,
        "hemoglobin_level": 9.5,
        "white_blood_cell_count": 11.2,
        "creatinine_level": 1.2,
        "systolic_bp": 145,
        "heart_rate": 88,
        "emergency_visits_last_30d": 1,
        "admission_type": "Emergency",
        "insurance_type": "Medicare"
    }
    
    try:
        response = requests.post(url, json=patient_data)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Prediction: {result['prediction']}")
            print(f"Readmission Risk: {result['readmission_risk']:.2%}")
            print(f"Risk Level: {result['risk_level']}")
            print(f"Recommendations: {result['recommendations'][:2]}...")
        else:
            print(f"Error: {response.text}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

# Test low risk patient
def test_low_risk_prediction():
    print("\n=== Testing Low Risk Patient ===")
    url = "http://localhost:5000/predict"
    
    patient_data = {
        "age": 35,
        "gender": "M",
        "length_of_stay_days": 2,
        "comorbidity_count": 0,
        "previous_admissions_90d": 0,
        "medications_count": 1,
        "hemoglobin_level": 14.5,
        "white_blood_cell_count": 6.5,
        "creatinine_level": 0.8,
        "systolic_bp": 118,
        "heart_rate": 72,
        "emergency_visits_last_30d": 0,
        "admission_type": "Elective",
        "insurance_type": "Private"
    }
    
    try:
        response = requests.post(url, json=patient_data)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Prediction: {result['prediction']}")
            print(f"Readmission Risk: {result['readmission_risk']:.2%}")
            print(f"Risk Level: {result['risk_level']}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

# Test batch prediction
def test_batch_prediction():
    print("\n=== Testing Batch Prediction ===")
    url = "http://localhost:5000/batch_predict"
    
    batch_data = {
        "patients": [
            {
                "age": 72, 
                "length_of_stay_days": 8, 
                "comorbidity_count": 4, 
                "admission_type": "Emergency",
                "previous_admissions_90d": 2,
                "medications_count": 5,
                "hemoglobin_level": 9.5,
                "white_blood_cell_count": 11.2,
                "creatinine_level": 1.2,
                "systolic_bp": 145,
                "heart_rate": 88,
                "emergency_visits_last_30d": 1,
                "insurance_type": "Medicare"
            },
            {
                "age": 45, 
                "length_of_stay_days": 2, 
                "comorbidity_count": 1, 
                "admission_type": "Elective",
                "previous_admissions_90d": 0,
                "medications_count": 2,
                "hemoglobin_level": 13.5,
                "white_blood_cell_count": 7.2,
                "creatinine_level": 0.9,
                "systolic_bp": 120,
                "heart_rate": 75,
                "emergency_visits_last_30d": 0,
                "insurance_type": "Private"
            },
            {
                "age": 85, 
                "length_of_stay_days": 12, 
                "comorbidity_count": 5, 
                "admission_type": "Emergency",
                "previous_admissions_90d": 3,
                "medications_count": 8,
                "hemoglobin_level": 8.5,
                "white_blood_cell_count": 14.5,
                "creatinine_level": 1.8,
                "systolic_bp": 160,
                "heart_rate": 95,
                "emergency_visits_last_30d": 2,
                "insurance_type": "Medicare"
            }
        ]
    }
    
    try:
        response = requests.post(url, json=batch_data)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Batch ID: {result['batch_id']}")
            print(f"Total Patients: {result['total_patients']}")
            print(f"Successful: {result['successful_predictions']}")
            for res in result['results'][:2]:
                if 'error' not in res:
                    print(f"  Patient {res['index']}: Risk = {res['risk_level']} ({res['readmission_risk']:.2%})")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

# Test model metrics
def test_metrics():
    print("\n=== Testing Model Metrics ===")
    try:
        response = requests.get("http://localhost:5000/model/metrics")
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            metrics = response.json()
            print(f"Total Predictions: {metrics.get('total_predictions', 0)}")
            if 'prediction_distribution' in metrics:
                print(f"Readmission Rate: {metrics['prediction_distribution'].get('readmission_rate', 0):.2%}")
            if 'performance_metrics' in metrics and 'accuracy' in metrics['performance_metrics']:
                print(f"Model Accuracy: {metrics['performance_metrics']['accuracy']:.2%}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

# Test invalid input
def test_invalid_input():
    print("\n=== Testing Invalid Input (Error Handling) ===")
    url = "http://localhost:5000/predict"
    
    # Missing required fields
    invalid_data = {
        "age": 72
        # Missing other required fields
    }
    
    try:
        response = requests.post(url, json=invalid_data)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 400:
            result = response.json()
            print(f"Error Response: {result.get('error', 'Unknown error')}")
        return response.status_code == 400
    except Exception as e:
        print(f"Error: {e}")
        return False

# Main test suite
def run_all_tests():
    print("=" * 50)
    print("Starting Healthcare ML API Tests")
    print("=" * 50)
    
    # First check if server is running
    print("\nChecking if server is running...")
    try:
        requests.get("http://localhost:5000/health", timeout=2)
    except:
        print("\n❌ ERROR: Server is not running!")
        print("Please start the server first with: python app.py")
        print("Then run this test script again.\n")
        return
    
    # Run all tests
    tests = [
        ("Health Check", test_health),
        ("Single Prediction", test_single_prediction),
        ("Low Risk Prediction", test_low_risk_prediction),
        ("Batch Prediction", test_batch_prediction),
        ("Model Metrics", test_metrics),
        ("Invalid Input Handling", test_invalid_input)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n--- Running {test_name} ---")
        try:
            result = test_func()
            results.append((test_name, result))
            print(f"✓ {test_name}: {'PASSED' if result else 'FAILED'}")
        except Exception as e:
            print(f"✗ {test_name}: ERROR - {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    passed = sum(1 for _, result in results if result)
    total = len(results)
    print(f"Tests Passed: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 All tests passed! API is working correctly.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the errors above.")
    
    print("\n💡 API Documentation:")
    print("  - POST /predict - Single prediction")
    print("  - POST /batch_predict - Batch predictions")
    print("  - GET /health - Health check")
    print("  - GET /model/metrics - Model metrics")
    print("  - GET / - Dashboard")
    print("\n📝 Example curl command:")
    print('  curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d \'{"age":72,"length_of_stay_days":8,"comorbidity_count":4,"admission_type":"Emergency"}\'')

if __name__ == "__main__":
    run_all_tests()