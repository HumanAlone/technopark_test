import pandas as pd

def prepare_features(df: pd.DataFrame, drop_columns: list = None):
    drop_columns = drop_columns or ["id", "diagnosed_diabetes"]
    return df.drop(columns=drop_columns, errors="ignore")