import pandas as pd
from catboost import CatBoostClassifier


def get_categorical_features(
    df: pd.DataFrame, target_col: str = "diagnosed_diabetes"
) -> list:
    """Автоматически определяем категориальные фичи"""
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    return [col for col in cat_cols if col != target_col]


def train_model(X_train: pd.DataFrame, y_train: pd.Series, cat_features: list):
    """Обучаем CatBoost-модель."""
    model = CatBoostClassifier(
        iterations=50,
        learning_rate=0.1,
        depth=4,
        verbose=False,
        random_state=42,
        cat_features=cat_features,
        thread_count=1,
        allow_writing_files=False,
    )
    model.fit(X_train, y_train)
    return model
