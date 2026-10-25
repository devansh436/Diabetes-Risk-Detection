"""Data loading + preprocessing helpers (Pima Indians Diabetes dataset)."""
import logging
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]
TARGET = "Outcome"
DATA_PATH = Path("data/diabetes.csv")
DATA_URL = (
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/"
    "pima-indians-diabetes.data.csv"
)
# In this dataset a 0 in these columns is a missing measurement, not a real value.
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def zeros_to_nan(X):
    """Replace impossible zeros with NaN (used inside the sklearn pipeline)."""
    X = pd.DataFrame(X, columns=FEATURES).astype(float).copy()
    X[ZERO_AS_MISSING] = X[ZERO_AS_MISSING].replace(0, np.nan)
    return X


def download_dataset(path: Path = DATA_PATH) -> Path:
    """Download the real dataset once and store it with a header row."""
    path.parent.mkdir(parents=True, exist_ok=True)
    raw, _ = urllib.request.urlretrieve(DATA_URL)
    df = pd.read_csv(raw, header=None, names=FEATURES + [TARGET])
    df.to_csv(path, index=False)
    return path


def make_synthetic(n: int = 800, seed: int = 42) -> pd.DataFrame:
    """Fallback dataset so the pipeline/tests still run without the real CSV."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "Pregnancies": rng.poisson(3.5, n).clip(0, 17),
        "Glucose": rng.normal(120, 30, n).clip(50, 200),
        "BloodPressure": rng.normal(70, 12, n).clip(30, 120),
        "SkinThickness": rng.normal(25, 10, n).clip(5, 60),
        "Insulin": rng.gamma(2.0, 60, n).clip(15, 600),
        "BMI": rng.normal(32, 6.5, n).clip(18, 60),
        "DiabetesPedigreeFunction": rng.gamma(2.0, 0.25, n).clip(0.08, 2.4),
        "Age": rng.gamma(6, 5.5, n).clip(21, 80),
    })
    z = (
        0.045 * (df["Glucose"] - 120) + 0.07 * (df["BMI"] - 32)
        + 0.03 * (df["Age"] - 33) + 0.9 * (df["DiabetesPedigreeFunction"] - 0.5) - 0.9
    )
    df[TARGET] = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    return df


def load_data(path: Path = DATA_PATH) -> tuple[pd.DataFrame, str]:
    """Return (dataframe, source) where source is 'csv' or 'synthetic'."""
    path = Path(path)
    if path.exists():
        df = pd.read_csv(path)
        if TARGET not in df.columns:  # headerless version of the file
            df = pd.read_csv(path, header=None, names=FEATURES + [TARGET])
        return df, "csv"
    log.warning("%s not found - using SYNTHETIC data. Run `python -m src.data` first.", path)
    return make_synthetic(), "synthetic"


if __name__ == "__main__":
    print(f"Saved dataset to {download_dataset()}")
