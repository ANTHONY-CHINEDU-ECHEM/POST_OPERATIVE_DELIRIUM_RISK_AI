# Postoperative Delirium Risk Prediction — EHR + Wearable Actigraphy

Predicts postoperative delirium (POD) risk in elderly surgical patients by
fusing pre-op EHR variables with a wrist-actigraphy-derived circadian
disruption signal, to trigger the HELP prevention bundle before anesthesia.

> **Reference-implementation note.** The briefing specifies an LSTM
> encoder over 7-day actigraphy traces feeding a LightGBM classifier. This
> repo's LSTM stand-in is a **lightweight statistical actigraphy encoder**
> (`src/actigraphy_encoder.py`) — sleep-efficiency, rest-activity rhythm
> amplitude, and fragmentation summary features computed directly from the
> synthetic minute-level traces — so the repo has no PyTorch dependency and
> trains in seconds. Swapping in a real LSTM/GGIR pipeline is a drop-in
> extension (see `docs/EXTENDING.md`).

## Quickstart

```bash
pip install -r requirements.txt
python data/generate_data.py            # synthetic EHR + actigraphy -> data/
python src/train.py                     # trains models/pod_lgbm.joblib
python src/evaluate.py                  # AUC-ROC, precision@top20%
python fairness/audit.py                # subgroup AUC/precision parity report
uvicorn serving.api:app --reload
```

## Structure

| Path | Purpose |
|---|---|
| `data/generate_data.py` | Synthesizes geriatric surgical episodes: EHR + minute-level actigraphy |
| `data/pod_episodes.csv` | Tabular EHR + actigraphy-summary dataset (generated, included) |
| `data/actigraphy_raw.npz` | Raw simulated minute-level actigraphy traces (generated, included) |
| `src/actigraphy_encoder.py` | Extracts sleep-efficiency / rest-activity-rhythm embedding from raw traces |
| `src/train.py` | Trains the LightGBM fusion classifier (tabular + actigraphy embedding) |
| `src/evaluate.py` | AUC-ROC, precision@top-20%-risk, ablation with/without actigraphy |
| `fairness/audit.py` | Subgroup AUC & selection-rate parity across age/sex/frailty |
| `serving/api.py` | FastAPI `/predict`, `/health` |
| `dashboard/app.py` | Streamlit pre-op huddle dashboard |

## License
MIT — portfolio/demonstration use.
