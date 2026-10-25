"""Training pipeline: load -> split -> preprocess+train -> evaluate -> quality gate -> save."""
import hashlib
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from src.data import FEATURES, TARGET, load_data, zeros_to_nan

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("train")

MIN_ROC_AUC = float(os.getenv("MIN_ROC_AUC", "0.70"))  # quality gate used by CI
# MIN_ROC_AUC = .99  # quality gate used by CI
PARAMS = {"n_estimators": 200, "max_depth": 6, "min_samples_leaf": 3, "random_state": 42}


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("zeros_to_nan", FunctionTransformer(zeros_to_nan)),
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", RandomForestClassifier(**PARAMS)),
    ])


def _git_sha() -> str:
    if os.getenv("GIT_SHA"):
        return os.environ["GIT_SHA"][:7]
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return "unknown"


def train(df: pd.DataFrame | None = None, out_dir: str = "models", source: str = "custom") -> dict:
    if df is None:
        df, source = load_data()
    X, y = df[FEATURES], df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    pipe = build_pipeline().fit(X_tr, y_tr)
    pred, proba = pipe.predict(X_te), pipe.predict_proba(X_te)[:, 1]
    metrics = {
        "accuracy": round(accuracy_score(y_te, pred), 4),
        "precision": round(precision_score(y_te, pred), 4),
        "recall": round(recall_score(y_te, pred), 4),
        "f1": round(f1_score(y_te, pred), 4),
        "roc_auc": round(roc_auc_score(y_te, proba), 4),
    }
    log.info("metrics: %s", metrics)

    if metrics["roc_auc"] < MIN_ROC_AUC:
        raise RuntimeError(f"Quality gate failed: roc_auc {metrics['roc_auc']} < {MIN_ROC_AUC}")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out / "model.joblib")
    meta = {
        "model_version": os.getenv("MODEL_VERSION", datetime.now(timezone.utc).strftime("%Y%m%d.%H%M")),
        "git_sha": _git_sha(),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "algorithm": "RandomForestClassifier",
        "params": PARAMS,
        "data_source": source,
        "data_rows": int(len(df)),
        "data_sha256": hashlib.sha256(pd.util.hash_pandas_object(df).values.tobytes()).hexdigest()[:12],
        "features": FEATURES,
        "metrics": metrics,
    }
    (out / "metrics.json").write_text(json.dumps(meta, indent=2))
    log.info("saved model v%s to %s", meta["model_version"], out)
    return meta


if __name__ == "__main__":
    try:
        train()
    except RuntimeError as e:
        log.error(e)
        sys.exit(1)
