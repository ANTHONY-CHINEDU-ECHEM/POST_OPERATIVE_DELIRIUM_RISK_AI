"""Subgroup fairness audit: AUC-ROC and top-20% selection rate stratified
by age band, sex, and frailty tercile. Stands in for the Fairlearn audit
referenced in the briefing — implemented directly with scikit-learn metrics
to avoid an extra heavy dependency.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from train import load_full, FEATURE_COLS

ROOT = Path(__file__).parent.parent


def subgroup_report(full, oof_col="oof_pred"):
    full = full.copy()
    full["age_band"] = pd.cut(full["age"], bins=[64, 72, 80, 96], labels=["65-72", "73-80", "81+"])
    full["frailty_tercile"] = pd.qcut(full["frailty_index"], 3, labels=["low", "mid", "high"])

    threshold = float(np.quantile(full[oof_col], 0.80))
    full["flagged"] = full[oof_col] >= threshold

    report = {"overall": {
        "auc": round(roc_auc_score(full["postop_delirium"], full[oof_col]), 4),
        "selection_rate": round(float(full["flagged"].mean()), 4),
        "n": int(len(full)),
    }}

    for dim in ["sex", "age_band", "frailty_tercile"]:
        report[dim] = {}
        for val, sub in full.groupby(dim, observed=True):
            if sub["postop_delirium"].nunique() < 2 or len(sub) < 30:
                auc = None
            else:
                auc = round(roc_auc_score(sub["postop_delirium"], sub[oof_col]), 4)
            report[dim][str(val)] = {
                "auc": auc,
                "selection_rate": round(float(sub["flagged"].mean()), 4),
                "n": int(len(sub)),
            }

    aucs = [v["auc"] for k in ["sex", "age_band", "frailty_tercile"] for v in report[k].values() if v["auc"] is not None]
    report["fairness_gap_max_minus_min_auc"] = round(max(aucs) - min(aucs), 4) if aucs else None
    return report


def main():
    oof_path = ROOT / "models" / "oof_predictions.csv"
    if not oof_path.exists():
        raise SystemExit("Run src/train.py first to produce out-of-fold predictions.")
    full = pd.read_csv(oof_path)
    report = subgroup_report(full)
    with open(ROOT / "fairness" / "fairness_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
