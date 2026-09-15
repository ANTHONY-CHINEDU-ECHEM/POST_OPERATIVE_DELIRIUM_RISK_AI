import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
from train import load_full, FEATURE_COLS

MODEL_PATH = ROOT / "models" / "pod_lgbm.joblib"


def test_data_shapes():
    full = load_full()
    assert len(full) > 1000
    assert set(FEATURE_COLS).issubset(full.columns)
    assert full["postop_delirium"].isin([0, 1]).all()


def test_incidence_is_clinically_plausible():
    full = load_full()
    rate = full["postop_delirium"].mean()
    assert 0.05 < rate < 0.55


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="Run src/train.py first")
def test_model_predicts_valid_probabilities():
    bundle = joblib.load(MODEL_PATH)
    full = load_full()
    preds = bundle["model"].predict_proba(full[FEATURE_COLS].head(50))[:, 1]
    assert (preds >= 0).all() and (preds <= 1).all()


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="Run src/train.py first")
def test_model_beats_random_baseline():
    from sklearn.metrics import roc_auc_score
    bundle = joblib.load(MODEL_PATH)
    full = load_full()
    preds = bundle["model"].predict_proba(full[FEATURE_COLS])[:, 1]
    auc = roc_auc_score(full["postop_delirium"], preds)
    assert auc > 0.65
