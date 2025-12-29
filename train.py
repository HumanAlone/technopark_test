import pandas as pd
from sklearn.model_selection import train_test_split
from src.data import load_data, clean_data, validate_data
from src.model import get_categorical_features, train_model
import mlflow
from mlflow import MlflowClient
import os

mlflow.set_tracking_uri("file://" + os.path.abspath("mlruns"))
mlflow.set_experiment("diabetes_prediction")

if __name__ == "__main__":
    df = load_data("data/train.csv")
    df = clean_data(df)
    df = validate_data(df)

    X = df.drop(columns=["diagnosed_diabetes", "id"])
    y = df["diagnosed_diabetes"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    cat_features = get_categorical_features(X)

    with mlflow.start_run() as run:
        model = train_model(X_train, y_train, cat_features)

        mlflow.log_params({
            "iterations": 200,
            "learning_rate": 0.1,
            "depth": 4,
            "n_train": len(X_train),
            "n_test": len(X_test)
        })

        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        mlflow.log_metrics({
            "train_accuracy": train_score,
            "test_accuracy": test_score
        })

        # Сохраняем и регистрируем модель
        mlflow.catboost.log_model(model, "model")
        model_name = "diabetes_model"
        mlflow.register_model(
            model_uri=f"runs:/{run.info.run_id}/model",
            name=model_name
        )

        # === Автоматически переводим в Production ===
        client = MlflowClient()
        # Получаем последнюю версию модели
        latest_version = client.get_latest_versions(model_name, stages=[]) [0].version
        # Переводим её в Production
        client.transition_model_version_stage(
            name=model_name,
            version=latest_version,
            stage="Production",
            archive_existing_versions=True  # если уже есть Production — архивируем
        )
        print(f"Модель версии {latest_version} помечена как Production")