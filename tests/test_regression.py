import os
from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).parent.parent / "data" / "train.csv"
TESTS_DIR = Path(__file__).parent
PROJECT_ROOT = TESTS_DIR.parent
os.environ["MLFLOW_TRACKING_URI"] = f"file://{PROJECT_ROOT}/mlruns"

import mlflow

model_name = "diabetes_model"
model = mlflow.pyfunc.load_model(f"models:/{model_name}@Production")


def test_no_regression():
    # Загружаем тестовые данные
    df = pd.read_csv(DATA_PATH, nrows=1000)

    X = df.drop(columns=["diagnosed_diabetes", "id"])
    y_true = df["diagnosed_diabetes"].astype(int)

    # Предсказания
    y_pred_proba = model.predict(X)
    y_pred = (y_pred_proba > 0.5).astype(int)

    from sklearn.metrics import accuracy_score

    acc = accuracy_score(y_true, y_pred)

    # Если accuracy < 0.6 — значит, что-то сломалось
    assert acc >= 0.6, f"Accuracy {acc} is too low — possible regression"
