# Postoperative Delirium Risk Prediction System
## AI-Driven Clinical Decision Support Using EHR Data and Wearable Actigraphy

<div align="center">

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)](https://www.python.org)
[![LightGBM](https://img.shields.io/badge/ML%20Framework-LightGBM-green?style=flat-square)](https://lightgbm.readthedocs.io/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009485?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)
[![Jupyter](https://img.shields.io/badge/Analysis-Jupyter-F37726?style=flat-square&logo=jupyter)](https://jupyter.org/)

</div>

---

## 📋 Table of Contents

- [Executive Summary](#executive-summary)
- [Clinical Motivation & Background](#clinical-motivation--background)
- [System Architecture](#system-architecture)
- [Key Features](#key-features)
- [Performance Metrics](#performance-metrics)
- [Installation & Quick Start](#installation--quick-start)
- [Repository Structure](#repository-structure)
- [Documentation](#documentation)
- [Usage Guide](#usage-guide)
- [Dashboard Preview](#dashboard-preview)
- [Model Architecture & Fairness](#model-architecture--fairness)
- [Extension & Deployment](#extension--deployment)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Executive Summary

This repository implements a **production-ready machine learning system** for predicting postoperative delirium (POD) risk in elderly surgical patients. By integrating preoperative electronic health record (EHR) data with circadian rhythm disturbance signals derived from 7-day wearable actigraphy, the system enables:

- **Early clinical intervention**: Automatic risk stratification to trigger the HELP (Hospital Elder Life Program) prevention bundle before anesthesia
- **Multimodal data fusion**: Combines tabular clinical variables with time-series actigraphy features via a statistical embedding encoder
- **Fairness-aware predictions**: Includes built-in audit infrastructure to detect and monitor bias across demographic subgroups
- **Clinical explainability**: SHAP integration for feature attribution and risk factor interpretation
- **Rapid inference**: FastAPI REST service and Streamlit dashboard for real-time decision support

**Technical Stack**: Python 3.9+, LightGBM classifier, NumPy/Pandas, FastAPI, Streamlit, SHAP, Scikit-learn

**Intended Use**: Pre-operative decision support system for geriatric surgical teams. **Not** a diagnostic tool.

---

## Clinical Motivation & Background

### The Problem
Postoperative delirium affects 10–50% of elderly surgical patients and is associated with:
- Prolonged hospital length of stay
- Increased mortality and morbidity
- Higher healthcare costs
- Functional decline and cognitive complications

Early identification of high-risk patients enables **preventive interventions before surgery**, improving outcomes significantly.

### The Opportunity
Recent evidence demonstrates that:
1. **Pre-operative sleep disruption** (measured via actigraphy) predicts delirium risk
2. **Multi-modal risk models** (clinical + actigraphy) outperform clinical variables alone
3. **Just-in-time prevention** (HELP bundle activation) can reduce POD incidence by 30–40%

This system operationalizes this science into a clinically deployable tool.

### Why Actigraphy + EHR?
- **Actigraphy** (wrist-worn accelerometer) is non-invasive and captures pre-op circadian disruption
- **EHR variables** encode frailty, comorbidity burden, and surgical risk in standardized form
- **Fusion** of both modalities provides ~8–9% additional AUC-ROC lift vs. clinical variables alone

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     Data Generation Layer                         │
│  Synthetic EHR cohort (4,200 geriatric episodes) + 7-day traces   │
│  (Extensible to real clinical data — see EXTENDING.md)           │
└────────────────────────────┬─────────────────────────────────────┘
                             │
        ┌────────────────────┴─────────────────────┐
        │                                          │
┌───────▼─────────────┐              ┌────────────▼──────────┐
│  Actigraphy Raw     │              │  EHR Variables         │
│  Traces (NPZ)       │              │  (Tabular CSV)         │
│  7-day×1440 min     │              │  4,200×47 features     │
└────────┬────────────┘              └──────────┬─────────────┘
         │                                      │
         │   Statistical Encoder                │
         │   (Sleep efficiency,                 │
         │    Rest-activity rhythm              │
         │    amplitude, fragmentation)         │
         │                                      │
         └──────────────┬──────────────────────┘
                        │
        ┌───────────────▼────────────────┐
        │  Feature Fusion                │
        │  (Clinical + Actigraphy)      │
        │  Engineered feature set        │
        └──────────────┬─────────────────┘
                       │
        ┌──────────────▼──────────────┐
        │   LightGBM Classifier       │
        │   5-fold CV ensemble        │
        │   Trained model artifact    │
        └────────┬────────────────────┘
                 │
     ┌───────────┴───────────┬──────────────┐
     │                       │              │
┌────▼──────┐       ┌────────▼────┐  ┌─────▼──────────┐
│ Evaluation│       │ Fairness    │  │ REST API       │
│ Metrics   │       │ Audit       │  │ (FastAPI)      │
│ (AUC, P@) │       │ (Subgroups) │  │ /predict       │
└───────────┘       └─────────────┘  └────────────────┘
                                              │
                                     ┌────────▼─────────┐
                                     │ Streamlit        │
                                     │ Dashboard        │
                                     │ (Pre-op huddle)  │
                                     └──────────────────┘
```

---

## Key Features

### 🔬 Multimodal Feature Fusion
- **EHR Features** (47 clinical variables): Age, comorbidity burden (Charlson index), frailty, renal function, anesthesia risk, surgery type, pre-op vitals
- **Actigraphy Features** (4 engineered embeddings): 
  - Sleep efficiency (7-day mean)
  - Rest-activity rhythm amplitude (circadian strength)
  - Fragmentation index (sleep-wake regularity)
  - Pre-op sleep timing variance

### 📊 Transparent Model Architecture
- **LightGBM gradient boosting** with 5-fold cross-validation
- **SHAP integration** for local and global feature importance
- **Ablation analysis**: Quantifies actigraphy contribution (8–9% AUC lift)

### ⚖️ Fairness & Bias Auditing
Automated subgroup performance analysis:
- **Age stratification** (65–75, 75–85, 85+)
- **Sex-based parity** (male/female)
- **Frailty terciles** (low/medium/high)
- Metrics: AUC-ROC, selection rate parity, precision@top-20%

### 🚀 Production-Ready Inference
- **FastAPI REST service** with health checks and batch prediction
- **Streamlit pre-op dashboard** for clinical huddle use
- **Sub-second latency** for real-time decisions
- **Model versioning** via joblib artifacts

### 📈 Comprehensive Evaluation
- Cross-validated AUC-ROC: **0.70** (typical; 0.67–0.72 across folds)
- Precision @ top-20%-risk threshold: **0.50**
- POD incidence (synthetic cohort): 26.2%

---

## Performance Metrics

### Model Performance
| Metric                      | Value          | Notes                                         |
|-----------------------------|----------------|-----------------------------------------------|
| **AUC-ROC (5-fold CV)**     | 0.700 (0.67–0.72) | Mean ± std of folds; typical for synthetic data |
| **Precision @ Top 20%**     | 0.50           | Of the 20% highest-risk patients, 50% develop POD |
| **Recall @ Top 20%**        | 0.38           | Captures 38% of all POD cases in top quintile |
| **Cohort Size**             | 4,200          | Simulated geriatric surgical episodes         |
| **POD Incidence**           | 26.2%          | Realistic for high-risk elderly population    |

### Feature Contribution
- **Actigraphy features** contribute ~8–9% AUC lift over clinical variables alone
- **Top predictors**: Age, Charlson comorbidity index, sleep fragmentation, surgery duration
- See `ablation_report.json` and SHAP outputs for detailed attribution

### Fairness Audit Results
See `fairness_report.json` after running `python fairness/audit.py`:
- **AUC parity**: Well-balanced across age/sex strata
- **Fairness gaps**: Minor AUC variance across frailty terciles (flagged for further investigation)
- Recommendation: Monitor subgroup performance closely during pilot deployment

---

## Installation & Quick Start

### Prerequisites
- Python 3.9 or later
- pip or conda
- ~2 GB disk space for synthetic data + model artifacts
- Unix-like environment (Linux, macOS) recommended; Windows requires Git Bash or WSL

### Installation

**Clone the repository:**
```bash
git clone https://github.com/ANTHONY-CHINEDU-ECHEM/POST_OPERATIVE_DELIRIUM_RISK_AI.git
cd POST_OPERATIVE_DELIRIUM_RISK_AI
```

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Verify installation:**
```bash
python -c "import lightgbm, fastapi, streamlit; print('All dependencies installed successfully')"
```

### Quickstart Pipeline

Run the complete end-to-end workflow in ~2 minutes:

```bash
# 1. Generate synthetic data
python data/generate_data.py

# 2. Train LightGBM model (model artifact → pod_lgbm.joblib)
python src/train.py

# 3. Evaluate model performance (metrics → metrics.json, ablation_report.json)
python src/evaluate.py

# 4. Audit fairness across demographic subgroups
python fairness/audit.py

# 5. Launch FastAPI service (default: http://localhost:8000)
uvicorn serving.api:app --reload

# 6. In a separate terminal, launch Streamlit dashboard (default: http://localhost:8501)
streamlit run serving/app.py
```

All outputs (model, data, reports) are written to the repository root or subdirectories.

---

## Repository Structure

```
POST_OPERATIVE_DELIRIUM_RISK_AI/
│
├── data/
│   ├── generate_data.py                 # Synthetic EHR + actigraphy generation
│   ├── pod_episodes.csv                 # Tabular dataset (4,200 × 47 + label)
│   └── actigraphy_sample_traces.npz     # Raw minute-level traces (4,200 × 10,080)
│
├── src/
│   ├── actigraphy_encoder.py            # Statistical actigraphy feature extraction
│   ├── train.py                         # LightGBM model training + cross-validation
│   └── evaluate.py                      # Performance evaluation + ablation analysis
│
├── fairness/
│   └── audit.py                         # Fairness audit: subgroup AUC/parity metrics
│
├── serving/
│   ├── api.py                           # FastAPI service (/predict, /health endpoints)
│   └── app.py                           # Streamlit pre-op huddle dashboard
│
├── notebooks/
│   └── eda.ipynb                        # Exploratory data analysis (Jupyter)
│
├── tests/
│   ├── test_pipeline.py                 # Unit tests for core modules
│   └── test_fairness.py (optional)      # Fairness module tests
│
├── docs/
│   ├── EXTENDING.md                     # Roadmap: real data, LSTM, deployment
│   ├── MODEL_CARD.md                    # Model specifications, metrics, limitations
│   └── API_SCHEMA.md (optional)         # OpenAPI endpoint reference
│
├── MODEL_CARD.md                        # Model card (intended use, limitations, fairness)
├── EXTENDING.md                         # Extension guidance for production use
├── requirements.txt                     # Dependency specifications
├── LICENSE                              # MIT License
├── README.md                            # This file
│
├── pod_lgbm.joblib                      # Trained model artifact (generated)
├── metrics.json                         # Performance metrics JSON (generated)
├── ablation_report.json                 # Feature ablation results (generated)
├── fairness_report.json                 # Fairness audit output (generated)
├── oof_predictions.csv                  # Out-of-fold predictions (generated)
│
└── POST OPERATIVE DASHBOARD.png         # Dashboard screenshot
```

---

## Documentation

### Core Documents

1. **[MODEL_CARD.md](MODEL_CARD.md)** — Formal model documentation
   - Intended use and limitations
   - Training data and cohort statistics
   - Performance metrics and benchmarks
   - Fairness audit notes

2. **[EXTENDING.md](EXTENDING.md)** — Production roadmap
   - Integrating real wearable actigraphy (via GGIR)
   - Swapping in a true LSTM sequence encoder
   - IRB/governance pathway for real clinical data
   - Deployment considerations (containerization, HL7/FHIR integration)

3. **Code Documentation** — Inline docstrings in all modules
   - Run `python -m pydoc src.train` for function signatures
   - See function headers in `src/train.py`, `src/actigraphy_encoder.py`, etc.

---

## Usage Guide

### 1. Data Generation

Generate synthetic EHR + actigraphy cohort:

```bash
python data/generate_data.py
```

**Output:**
- `pod_episodes.csv` — 4,200 rows × 47 clinical features + POD label
- `actigraphy_sample_traces.npz` — Raw minute-level activity counts

**Customize:**
Edit `data/generate_data.py` to modify:
- Cohort size (default: 4,200)
- POD incidence (default: 26.2%)
- Actigraphy trace length (default: 7 days × 1,440 min)

### 2. Model Training

Train LightGBM classifier on fused features:

```bash
python src/train.py
```

**Output:**
- `pod_lgbm.joblib` — Serialized model (ready for inference)
- Cross-validation fold predictions (logged to stdout)

**Key parameters** (edit in `src/train.py`):
- `num_leaves`: LightGBM tree complexity (default: 31)
- `learning_rate`: Gradient boosting step size (default: 0.05)
- `n_estimators`: Number of boosting rounds (default: 500)

### 3. Evaluation & Metrics

Compute performance metrics and ablation analysis:

```bash
python src/evaluate.py
```

**Output:**
- `metrics.json` — AUC-ROC, precision@top-20%, thresholds
- `ablation_report.json` — Actigraphy vs. clinical-only model performance
- Detailed logs to stdout

### 4. Fairness Auditing

Run subgroup parity analysis:

```bash
python fairness/audit.py
```

**Output:**
- `fairness_report.json` — AUC, selection rates, parity metrics by age/sex/frailty
- Printout highlighting any fairness gaps

**Interpretation:**
- AUC difference < 5% across strata: acceptable
- Selection-rate parity: ensure top-20% selection is evenly distributed
- Frailty gaps: investigate further before real deployment

### 5. Real-Time Inference

#### Option A: FastAPI Service

Start the server:
```bash
uvicorn serving.api:app --reload
```

**Endpoints:**

**POST `/predict`** — Batch prediction
```json
{
  "episodes": [
    {
      "age": 75,
      "charlson_index": 2,
      "renal_function_stage": 2,
      "sleep_efficiency": 0.72,
      "rest_activity_amplitude": 0.65,
      "sleep_fragmentation": 0.42,
      ...
    }
  ]
}
```

**Response:**
```json
{
  "predictions": [
    {
      "pod_risk_score": 0.68,
      "high_risk_flag": true,
      "confidence_interval": [0.61, 0.74]
    }
  ]
}
```

**GET `/health`** — Service status

**Full API docs:** http://localhost:8000/docs (Swagger UI)

#### Option B: Streamlit Dashboard

Launch interactive dashboard:
```bash
streamlit run serving/app.py
```

**Features:**
- Patient lookup by ID
- Risk visualization (risk gauge + risk factors)
- Cohort-level statistics
- Model performance overview
- Batch patient upload

See [Dashboard Preview](#dashboard-preview) below.

### 6. Exploratory Analysis

Interactive Jupyter notebook:
```bash
jupyter notebook notebooks/eda.ipynb
```

Includes:
- Cohort demographics
- Feature distributions
- POD incidence by risk strata
- Actigraphy patterns in high-risk patients

---

## Dashboard Preview

![Post-Operative Delirium Risk Dashboard](POST%20OPERATIVE%20DASHBOARD.png)

**Dashboard Features:**
- **Risk Gauge**: Patient-level POD risk score (0–100)
- **High-Risk Alerts**: Automatic flagging for scores > 61.4% (top-20% threshold)
- **Contributing Factors**: SHAP-based breakdown of top predictive features
- **Cohort Overview**: Aggregate statistics, distribution plots
- **Model Performance**: AUC-ROC curves, calibration plots
- **Comparison Cohorts**: Stratified performance by age/sex/frailty

---

## Model Architecture & Fairness

### Model Flow

```
                    ┌─────────────────────────────────────┐
                    │  Raw Actigraphy Traces              │
                    │  (7 days × 1,440 minutes × 4,200)  │
                    └────────────┬────────────────────────┘
                                 │
                    ┌────────────▼─────────────────┐
                    │ Actigraphy Encoder           │
                    │ - Sleep efficiency           │
                    │ - RA rhythm amplitude        │
                    │ - Fragmentation index        │
                    │ - Sleep timing variance      │
                    └───────────┬──────────────────┘
                                │
        ┌───────────────────────┴──────────────────────┐
        │                                              │
   ┌────▼──────────────┐              ┌──────────────▼──┐
   │ Actigraphy        │              │ EHR Variables    │
   │ Features (4)      │              │ (43 clinical)    │
   └────┬──────────────┘              └────────┬─────────┘
        │                                      │
        └──────────────┬───────────────────────┘
                       │
           ┌───────────▼──────────────┐
           │ Feature Engineering      │
           │ (Scaling, interactions)  │
           └───────────┬──────────────┘
                       │
           ┌───────────▼──────────────────────┐
           │ LightGBM Classifier              │
           │ - 5-fold CV                      │
           │ - 500 boosting rounds            │
           │ - Max depth: 10, num_leaves: 31  │
           └───────────┬──────────────────────┘
                       │
        ┌──────────────┴──────────────┐
        │                             │
   ┌────▼──────────┐        ┌────────▼──────┐
   │ POD Risk      │        │ Feature       │
   │ Probability   │        │ Importance    │
   │ (0 to 1)      │        │ (SHAP values) │
   └───────────────┘        └───────────────┘
```

### Fairness Strategy

**Three-tier audit:**

1. **Stratification Variables**: Age band (3 levels), sex (2 levels), frailty (3 levels) → 18 subgroups
2. **Metrics**: AUC-ROC, precision@top-20%, selection-rate parity
3. **Action Thresholds**:
   - AUC difference > 10%: Investigate and document
   - Selection-rate imbalance > 15%: Retrain with fairness constraints
   - Consistent gaps in frailty: Risk adjustment before deployment

**Current Status** (synthetic cohort):
- ✅ Age/sex parity: Balanced (AUC < 5% difference)
- ⚠️ Frailty gaps: Minor disparities flagged; recommend real-data validation

---

## Extension & Deployment

### Near-Term Enhancements (see EXTENDING.md)

1. **Real Actigraphy Integration**
   - Replace synthetic traces with GGIR-processed wearable data
   - Validate on real pre-op patients (IRB approval required)
   - Expected AUC improvement: 0.70 → 0.75–0.80

2. **LSTM Sequence Encoder**
   - Replace statistical features with PyTorch LSTM on 7-day traces
   - End-to-end training with LightGBM head
   - ~3–5% additional AUC lift expected

3. **Production Deployment**
   - Docker containerization (api.py)
   - HL7/FHIR integration with EHR order entry
   - Automated fairness auditing in CI/CD pipeline
   - Model governance: version control, A/B testing framework

4. **Governance & Ethics**
   - IRB approval for real patient data
   - Informed consent & de-identification protocols
   - Bias monitoring dashboard (continuous fairness tracking)
   - Outcome tracking (POD incidence post-implementation)

---

## Testing

Run the test suite:

```bash
pytest tests/ -v
```

**Coverage:**
- Data generation and shape validation
- Actigraphy encoder outputs (4 features per episode)
- Model training and cross-validation
- API endpoint functionality
- Fairness audit subgroup stratification

---

## Requirements & Dependencies

| Package       | Version    | Purpose                                    |
|---------------|------------|--------------------------------------------|
| numpy         | ≥1.26      | Numerical computing, array handling       |
| pandas        | ≥2.0       | Data manipulation and analysis            |
| scikit-learn  | ≥1.3       | Preprocessing, metrics                    |
| lightgbm      | ≥4.0       | Gradient boosting classifier              |
| shap          | ≥0.44      | Feature importance & explainability       |
| joblib        | ≥1.3       | Model serialization                       |
| fastapi       | ≥0.110     | REST API framework                        |
| uvicorn       | ≥0.29      | ASGI server for FastAPI                   |
| streamlit     | (optional) | Interactive dashboard                     |
| pytest        | ≥8.0       | Unit testing                              |

Full list: [requirements.txt](requirements.txt)

---

## Contributing

We welcome contributions! Please:

1. **Fork** this repository
2. **Create a feature branch** (`git checkout -b feature/your-feature`)
3. **Commit** your changes with clear messages
4. **Push** to the branch and **open a Pull Request**

**Guidelines:**
- Follow PEP 8 style (use `black` for formatting)
- Add docstrings to all functions
- Include unit tests for new modules
- Update this README if adding major features

---

## Citation

If you use this work in research or clinical settings, please cite:

```bibtex
@software{pod_risk_ai_2024,
  title={Postoperative Delirium Risk Prediction System: EHR + Wearable Actigraphy Fusion},
  author={Echem, Anthony Chinedu},
  year={2024},
  url={https://github.com/ANTHONY-CHINEDU-ECHEM/POST_OPERATIVE_DELIRIUM_RISK_AI}
}
```

---

## License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) file for details.

**Summary**: Free use for research, education, and commercial purposes with attribution. No warranty implied.

---

## Acknowledgments

### Scientific Foundation
- HELP (Hospital Elder Life Program) prevention framework
- Recent literature on actigraphy and delirium risk
- Fairness in clinical AI best practices (FDA, WHO guidelines)

### Technical Stack
- LightGBM team for robust gradient boosting
- Plotly/Streamlit communities for visualization
- SHAP authors for interpretability framework
- FastAPI team for modern Python APIs

### Contributors
- Anthony Chinedu Echem (primary author)

---

## Support & Contact

### Troubleshooting

**Q: Model training is slow**
- A: Reduce `n_estimators` in `src/train.py` or cohort size in `data/generate_data.py`

**Q: "ImportError: No module named 'lightgbm'"**
- A: Run `pip install lightgbm` or re-run `pip install -r requirements.txt`

**Q: FastAPI service won't start**
- A: Ensure port 8000 is available; use `uvicorn serving.api:app --port 8001` to specify a different port

**Q: Fairness report shows large AUC gaps**
- A: This is expected on synthetic data; validate on real clinical cohort (see EXTENDING.md)

### Getting Help
- Check existing [GitHub Issues](https://github.com/ANTHONY-CHINEDU-ECHEM/POST_OPERATIVE_DELIRIUM_RISK_AI/issues)
- Review [EXTENDING.md](EXTENDING.md) for deployment questions
- See [MODEL_CARD.md](MODEL_CARD.md) for model-specific details

---

## Disclaimer

**This is a reference implementation for research and educational purposes.** Clinical deployment requires:
- ✅ Institutional Review Board (IRB) approval
- ✅ Clinical validation on real patient cohorts
- ✅ Regulatory compliance (FDA 510(k) if applicable)
- ✅ Clinical governance and outcome tracking

**Not a substitute for clinical judgment.** Model outputs are decision-support tools only.

---

<div align="center">

**Last Updated**: 2024  
**Status**: Active Development  
**Questions?** Open a GitHub Issue or check [EXTENDING.md](EXTENDING.md)

</div>
