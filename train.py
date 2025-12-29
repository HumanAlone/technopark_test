import os

import mlflow
import pandas as pd
from mlflow import MlflowClient
from sklearn.model_selection import train_test_split

from src.data import clean_data, load_data, validate_data
from src.model import get_categorical_features, train_model

mlflow.set_tracking_uri("mlruns")
mlflow.set_experiment("diabetes_prediction")

if __name__ == "__main__":
    df = load_data("data/train.csv")
    df = df.sample(n=5000, random_state=42).reset_index(drop=True)
    df = clean_data(df)
    df = validate_data(df)

    X = df.drop(columns=["diagnosed_diabetes", "id"])
    y = df["diagnosed_diabetes"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    cat_features = get_categorical_features(X)

    with mlflow.start_run() as run:
        model = train_model(X_train, y_train, cat_features)

        mlflow.log_params(
            {
                "iterations": 200,
                "learning_rate": 0.1,
                "depth": 4,
                "n_train": len(X_train),
                "n_test": len(X_test),
            }
        )

        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        mlflow.log_metrics({"train_accuracy": train_score, "test_accuracy": test_score})

        # Сохраняем и регистрируем модель
        mlflow.catboost.log_model(model, "model")
        model_name = "diabetes_model"
        mlflow.register_model(
            model_uri=f"runs:/{run.info.run_id}/model", name=model_name
        )

        client = MlflowClient()
        latest_version = client.search_model_versions(f"name='{model_name}'")[0].version
        # Устанавливаем алиас "Production"
        client.set_registered_model_alias(
            name=model_name, alias="Production", version=latest_version
        )
        print(f"Модель версии {latest_version} помечена алиасом 'Production'")
