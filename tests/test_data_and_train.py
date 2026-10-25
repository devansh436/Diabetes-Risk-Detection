import json

import numpy as np
import pandas as pd
import pytest

from src.data import FEATURES, TARGET, load_data, make_synthetic, zeros_to_nan
from src.train import train


# ---- unit tests -------------------------------------------------------------
def test_zeros_become_nan_only_in_medical_columns():
    df = pd.DataFrame([[0, 0, 0, 0, 0, 0, 0.5, 30]], columns=FEATURES)
    out = zeros_to_nan(df)
    assert out["Glucose"].isna().all() and out["BMI"].isna().all()
    assert out["Pregnancies"].iloc[0] == 0 and not np.isnan(out["Age"].iloc[0])


def test_synthetic_data_shape_and_target():
    df = make_synthetic(n=200)
    assert list(df.columns) == FEATURES + [TARGET]
    assert set(df[TARGET].unique()) == {0, 1}


def test_load_data_falls_back_to_synthetic(tmp_path):
    df, source = load_data(tmp_path / "missing.csv")
    assert source == "synthetic" and len(df) > 0


def test_load_data_reads_csv(tmp_path):
    p = tmp_path / "d.csv"
    make_synthetic(n=50).to_csv(p, index=False)
    df, source = load_data(p)
    assert source == "csv" and len(df) == 50


# ---- training pipeline (integration) -----------------------------------------
def test_train_writes_model_and_metadata(model_dir):
    assert (model_dir / "model.joblib").exists()
    meta = json.loads((model_dir / "metrics.json").read_text())
    assert {"model_version", "git_sha", "data_sha256", "metrics"} <= meta.keys()
    assert meta["metrics"]["roc_auc"] >= 0.70


def test_quality_gate_blocks_bad_model(tmp_path, monkeypatch):
    monkeypatch.setattr("src.train.MIN_ROC_AUC", 0.999)
    with pytest.raises(RuntimeError, match="Quality gate"):
        train(df=make_synthetic(seed=1), out_dir=str(tmp_path))
