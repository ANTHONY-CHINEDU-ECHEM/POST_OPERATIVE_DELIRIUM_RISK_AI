"""Evaluation report including the ablation study: tabular-only vs.
tabular+actigraphy, to validate the wearable-data contribution claimed in
the portfolio briefing (target: >=8% AUC lift).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from train import load_full, TABULAR_COLS, ACTG_COLS, FEATURE_COLS

ROOT = Path(__file__).parent.parent


def cv_auc(X, y, seed=42):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    aucs = []
    for tr, te in skf.split(X, y):
        m = LGBMClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                            subsample=0.85, colsample_bytree=0.85,
                            class_weight="balanced", random_state=seed, verbosity=-1)
        m.fit(X.iloc[tr], y.iloc[tr])
        p = m.predict_proba(X.iloc[te])[:, 1]
        aucs.append(roc_auc_score(y.iloc[te], p))
    return float(np.mean(aucs))


def main():
    full = load_full()
    y = full["postop_delirium"]

    auc_tabular_only = cv_auc(full[TABULAR_COLS], y)
    auc_full = cv_auc(full[FEATURE_COLS], y)
    lift_pct = (auc_full - auc_tabular_only) / auc_tabular_only * 100

    report = {
        "auc_tabular_only": round(auc_tabular_only, 4),
        "auc_tabular_plus_actigraphy": round(auc_full, 4),
        "actigraphy_auc_lift_pct": round(lift_pct, 2),
        "meets_8pct_lift_target": bool(lift_pct >= 8.0),
    }
    with open(ROOT / "models" / "ablation_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
