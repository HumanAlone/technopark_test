import os
from pathlib import Path

TESTS_DIR = Path(__file__).parent
PROJECT_ROOT = TESTS_DIR.parent
os.environ["MLFLOW_TRACKING_URI"] = f"file://{PROJECT_ROOT}/mlruns"

import mlflow
import pandas as pd


model_name = "diabetes_model"
model = mlflow.pyfunc.load_model(f"models:/{model_name}@Production")


def test_model_input_output():
    # Создаём DataFrame с одним примером
    input_data = pd.DataFrame(
        [
            {
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
        ]
    )

    output = model.predict(input_data)

    # Проверяем диапазон
    assert len(output) == 1
    proba = float(output[0])
    assert 0.0 <= proba <= 1.0
