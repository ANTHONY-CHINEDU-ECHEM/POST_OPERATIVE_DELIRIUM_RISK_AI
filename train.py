"""Train the LightGBM POD-risk fusion classifier (tabular EHR + actigraphy
embedding), with 5-fold stratified CV and a threshold tuned to flag the
top ~20% highest-risk patients (matched to typical HELP-bundle capacity).
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_score

sys.path.insert(0, str(Path(__file__).parent))
from actigraphy_encoder import encode_batch

ROOT = Path(__file__).parent.parent
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

TABULAR_COLS = ["age", "asa_class", "frailty_index", "baseline_mmse", "acb_score",
                "intraop_duration_min", "blood_loss_ml", "anesthesia_general"]
ACTG_COLS = ["actg_amplitude", "actg_interdaily_stability", "actg_intradaily_variability",
             "actg_sleep_efficiency", "actg_night_mean_activity"]
FEATURE_COLS = TABULAR_COLS + ACTG_COLS


def load_full():
    """Actigraphy summary features are precomputed into pod_episodes.csv by
    data/generate_data.py (one row per episode); only a small sample of raw
    minute-level traces ships separately (data/actigraphy_sample_traces.npz)
    for notebook/dashboard visualization, to keep the repo lightweight.
    """
    full = pd.read_csv(ROOT / "data" / "pod_episodes.csv")
    return full


def main():
    full = load_full()
    X = full[FEATURE_COLS]
    y = full["postop_delirium"]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    oof_pred = np.zeros(len(y))
    for train_idx, test_idx in skf.split(X, y):
        m = LGBMClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                            subsample=0.85, colsample_bytree=0.85,
                            class_weight="balanced", random_state=42, verbosity=-1)
        m.fit(X.iloc[train_idx], y.iloc[train_idx])
        p = m.predict_proba(X.iloc[test_idx])[:, 1]
        oof_pred[test_idx] = p
        aucs.append(roc_auc_score(y.iloc[test_idx], p))

    cv_auc = float(np.mean(aucs))

    # Final model on all data for deployment
    final_model = LGBMClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                  subsample=0.85, colsample_bytree=0.85,
                                  class_weight="balanced", random_state=42, verbosity=-1)
    final_model.fit(X, y)

    threshold = float(np.quantile(oof_pred, 0.80))  # top 20% flagged
    flagged = oof_pred >= threshold
    precision_top20 = precision_score(y, flagged)

    joblib.dump({"model": final_model, "feature_cols": FEATURE_COLS, "threshold": threshold},
                MODEL_DIR / "pod_lgbm.joblib")
    full.assign(oof_pred=oof_pred).to_csv(MODEL_DIR / "oof_predictions.csv", index=False)

    metrics = {"cv_auc_roc_mean": round(cv_auc, 4), "cv_auc_roc_folds": [round(a, 4) for a in aucs],
               "top20pct_threshold": round(threshold, 4), "precision_at_top20pct": round(float(precision_top20), 4),
               "n_episodes": int(len(y)), "pod_incidence": round(float(y.mean()), 4)}
    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
