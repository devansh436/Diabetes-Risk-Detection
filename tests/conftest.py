import pytest
from fastapi.testclient import TestClient

from src.data import make_synthetic
from src.train import train


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("models")
    train(df=make_synthetic(seed=0), out_dir=str(out), source="synthetic")
    return out


@pytest.fixture(scope="session")
def client(model_dir):
    import os

    os.environ["MODEL_PATH"] = str(model_dir / "model.joblib")
    from app.main import app

    with TestClient(app) as c:
        yield c


VALID = {
    "Pregnancies": 2, "Glucose": 120, "BloodPressure": 70, "SkinThickness": 20,
    "Insulin": 80, "BMI": 28.5, "DiabetesPedigreeFunction": 0.4, "Age": 35,
}
