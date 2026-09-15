"""Streamlit pre-op huddle dashboard. Run: streamlit run dashboard/app.py"""
import sys
from pathlib import Path

import joblib
import pandas as pd
import shap
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from train import load_full, FEATURE_COLS

ROOT = Path(__file__).parent.parent
st.set_page_config(page_title="Pre-Op Huddle — POD Risk", layout="wide")
st.title("🧠 Postoperative Delirium — Pre-Op Huddle Dashboard")

bundle = joblib.load(ROOT / "models" / "pod_lgbm.joblib")
model = bundle["model"]
explainer = shap.TreeExplainer(model)
full = load_full()

st.caption("Raw actigraphy trace preview is available for a 250-episode demo sample; risk scoring works for all episodes.")
episode_id = st.selectbox("Select patient episode", sorted(full["episode_id"].unique()))
row = full[full["episode_id"] == episode_id]
x = row[FEATURE_COLS]
p = float(model.predict_proba(x)[0, 1])

col1, col2 = st.columns([1, 2])
with col1:
    st.metric("POD risk", f"{p:.1%}")
    st.write("🔴 **Flag for HELP bundle**" if p >= bundle["threshold"] else "🟢 Standard care pathway")
    st.write(row[["age", "asa_class", "frailty_index", "acb_score", "actg_sleep_efficiency"]].T)

with col2:
    sv = explainer.shap_values(x)
    contrib = sorted(zip(FEATURE_COLS, sv[0]), key=lambda t: abs(t[1]), reverse=True)
    st.write("**Contributing factors:**")
    st.table(pd.DataFrame(contrib, columns=["feature", "shap_value"]))
