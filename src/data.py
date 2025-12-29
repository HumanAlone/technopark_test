import pandas as pd
from pathlib import Path

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Очистка: удаление дубликатов."""
    original_len = len(df)
    
    # Удаляем полные дубликаты
    df = df.drop_duplicates().reset_index(drop=True)
    
    # Удаляем дубликаты по id (оставляем первую запись)
    if "id" in df.columns:
        df = df.drop_duplicates(subset=["id"]).reset_index(drop=True)
    
    cleaned_len = len(df)
    if original_len != cleaned_len:
        print(f"Очистка: удалено {original_len - cleaned_len} дубликатов")
    return df

def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """Minimal validation: just check required columns."""
    required = {
        "age", "bmi", "gender", "diagnosed_diabetes"
    }
    assert required.issubset(df.columns), f"Missing columns: {required - set(df.columns)}"
    assert set(df["diagnosed_diabetes"].dropna().unique()).issubset({0.0, 1.0}), "Target must be binary"
    return df