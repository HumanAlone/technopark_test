from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_predict_endpoint():
    payload = {
        "age": 45,
        "alcohol_consumption_per_week": 2,
        "physical_activity_minutes_per_week": 150,
        "diet_score": 7.5,
        "sleep_hours_per_day": 7.0,
        "screen_time_hours_per_day": 4.0,
        "bmi": 28.5,
        "waist_to_hip_ratio": 0.9,
        "systolic_bp": 120,
        "diastolic_bp": 80,
        "heart_rate": 72,
        "cholesterol_total": 200,
        "hdl_cholesterol": 50,
        "ldl_cholesterol": 130,
        "triglycerides": 150,
        "gender": "Male",
        "ethnicity": "Caucasian",
        "education_level": "Bachelor",
        "income_level": "Middle",
        "smoking_status": "Never",
        "employment_status": "Employed",
        "family_history_diabetes": 1,
        "hypertension_history": 0,
        "cardiovascular_history": 0,
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "diabetes_probability" in data
    proba = data["diabetes_probability"]
    assert isinstance(proba, float)
    assert 0.0 <= proba <= 1.0
