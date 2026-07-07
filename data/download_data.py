"""
download_data.py
Downloads the Heart Disease UCI dataset and saves it to data/heart.csv
Run: python data/download_data.py
"""

import os
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__))
OUTPUT_PATH = os.path.join(DATA_DIR, "heart.csv")


def download_from_uci():
    """Download via ucimlrepo package (preferred)."""
    from ucimlrepo import fetch_ucirepo
    print("Fetching Heart Disease dataset from UCI ML Repository...")
    heart_disease = fetch_ucirepo(id=45)
    X = heart_disease.data.features
    y = heart_disease.data.targets
    df = pd.concat([X, y], axis=1)
    # Rename target to 'target' and binarise (0 = no disease, 1 = disease)
    target_col = df.columns[-1]
    df.rename(columns={target_col: "target"}, inplace=True)
    df["target"] = (df["target"] > 0).astype(int)
    return df


def download_fallback():
    """Fallback: download raw Cleveland data directly from UCI archive."""
    import urllib.request
    url = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/"
        "heart-disease/processed.cleveland.data"
    )
    columns = [
        "age", "sex", "cp", "trestbps", "chol", "fbs",
        "restecg", "thalach", "exang", "oldpeak", "slope",
        "ca", "thal", "target",
    ]
    print(f"Downloading from {url} ...")
    urllib.request.urlretrieve(url, "/tmp/cleveland.data")
    df = pd.read_csv("/tmp/cleveland.data", names=columns, na_values="?")
    df["target"] = (df["target"] > 0).astype(int)
    return df


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    try:
        df = download_from_uci()
    except Exception as e:
        print(f"ucimlrepo failed ({e}), trying direct download...")
        df = download_fallback()

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Dataset saved to {OUTPUT_PATH}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Target distribution:\n{df['target'].value_counts()}")
    print(f"Missing values:\n{df.isnull().sum()}")