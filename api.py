"""FastAPI inference service for the postoperative-delirium risk model."""
import sys
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from train import FEATURE_COLS

MODEL_DIR = Path(__file__).parent.parent / "models"
app = FastAPI(title="Postoperative Delirium Risk API", version="1.0.0")

_bundle = None


def _load():
    global _bundle
    if _bundle is None:
        _bundle = joblib.load(MODEL_DIR / "pod_lgbm.joblib")
    return _bundle


class PODFeatures(BaseModel):
    age: float
    asa_class: int
    frailty_index: float
    baseline_mmse: float
    acb_score: int
    intraop_duration_min: float
    blood_loss_ml: float
    anesthesia_general: int
    actg_amplitude: float
    actg_interdaily_stability: float
    actg_intradaily_variability: float
    actg_sleep_efficiency: float
    actg_night_mean_activity: float


class PredictionResponse(BaseModel):
    pod_risk_probability: float
    flagged_for_help_bundle: bool


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(x: PODFeatures):
    bundle = _load()
    row = pd.DataFrame([x.dict()])[FEATURE_COLS]
    p = float(bundle["model"].predict_proba(row)[0, 1])
    return PredictionResponse(pod_risk_probability=round(p, 4), flagged_for_help_bundle=p >= bundle["threshold"])
