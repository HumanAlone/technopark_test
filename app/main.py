import os
from pathlib import Path

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Получаем абсолютный путь к mlruns относительно корня проекта
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
MLRUNS_PATH = PROJECT_ROOT / "mlruns"

# Устанавливаем URI
os.environ["MLFLOW_TRACKING_URI"] = f"file://{MLRUNS_PATH}"

model_name = "diabetes_model"
model = mlflow.pyfunc.load_model(f"models:/{model_name}@Production")

app = FastAPI(title="Diabetes Prediction API")


class PredictionInput(BaseModel):
    age: int
    alcohol_consumption_per_week: int
    physical_activity_minutes_per_week: int
    diet_score: float
    sleep_hours_per_day: float
    screen_time_hours_per_day: float
    bmi: float
    waist_to_hip_ratio: float
    systolic_bp: int
    diastolic_bp: int
    heart_rate: int
    cholesterol_total: int
    hdl_cholesterol: int
    ldl_cholesterol: int
    triglycerides: int
    gender: str
    ethnicity: str
    education_level: str
    income_level: str
    smoking_status: str
    employment_status: str
    family_history_diabetes: int
    hypertension_history: int
    cardiovascular_history: int


@app.post("/predict")
def predict(input_data: PredictionInput):
    try:
        df = pd.DataFrame([input_data.dict()])
        proba = model.predict(df)[0]
        return {"diabetes_probability": float(proba)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model error: {str(e)}")
