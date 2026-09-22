# Postoperative Delirium Risk Prediction System

## AI Driven Clinical Decision Support Using EHR Data and Wearable Actigraphy

<div align="center">

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)](https://www.python.org)
[![LightGBM](https://img.shields.io/badge/ML%20Framework-LightGBM-green?style=flat-square)](https://lightgbm.readthedocs.io/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009485?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)
[![Jupyter](https://img.shields.io/badge/Analysis-Jupyter-F37726?style=flat-square&logo=jupyter)](https://jupyter.org/)

</div>

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Clinical Motivation and Background](#clinical-motivation-and-background)
3. [System Architecture](#system-architecture)
4. [Key Features](#key-features)
5. [Performance Metrics](#performance-metrics)
6. [Installation and Quick Start](#installation-and-quick-start)
7. [Repository Structure](#repository-structure)
8. [Documentation](#documentation)
9. [Usage Guide](#usage-guide)
10. [Dashboard Preview](#dashboard-preview)
11. [Model Architecture and Fairness](#model-architecture-and-fairness)
12. [Extension and Deployment](#extension-and-deployment)
13. [Testing](#testing)
14. [Requirements and Dependencies](#requirements-and-dependencies)
15. [Contributing](#contributing)
16. [Citation](#citation)
17. [License](#license)
18. [Acknowledgments](#acknowledgments)
19. [Support and Contact](#support-and-contact)
20. [Disclaimer](#disclaimer)

---

## Executive Summary

This repository implements a production ready machine learning system for predicting postoperative delirium risk in elderly surgical patients. By integrating preoperative electronic health record data with circadian rhythm disturbance signals derived from 7 day wearable actigraphy, the system enables:

- **Early clinical intervention.** Automatic risk stratification triggers the HELP (Hospital Elder Life Program) prevention bundle before anesthesia.
- **Multimodal data fusion.** Tabular clinical variables are combined with time series actigraphy features through a statistical embedding encoder.
- **Fairness aware predictions.** Built in audit infrastructure detects and monitors bias across demographic subgroups.
- **Clinical explainability.** SHAP integration provides feature attribution and risk factor interpretation.
- **Rapid inference.** A FastAPI REST service and a Streamlit dashboard support real time decision making.

**Technical stack:** Python 3.9 or later, a LightGBM classifier, NumPy and Pandas, FastAPI, Streamlit, SHAP, and scikit-learn.

**Intended use:** a pre operative decision support system for geriatric surgical teams. This is not a diagnostic tool.

---

## Clinical Motivation and Background

### The Problem

Postoperative delirium affects 10 to 50 percent of elderly surgical patients and is associated with:

- Prolonged hospital length of stay
- Increased mortality and morbidity
- Higher healthcare costs
- Functional decline and cognitive complications

Early identification of high risk patients enables preventive interventions before surgery, which can significantly improve outcomes.

### The Opportunity

Recent evidence demonstrates that:

1. Pre operative sleep disruption, as measured through actigraphy, predicts delirium risk.
2. Multimodal risk models, combining clinical and actigraphy data, outperform clinical variables alone.
3. Just in time prevention, through HELP bundle activation, can reduce postoperative delirium incidence by 30 to 40 percent.

This system operationalizes that science into a clinically deployable tool.

### Why Combine Actigraphy and EHR Data

- **Actigraphy**, using a wrist worn accelerometer, is non invasive and captures pre operative circadian disruption.
- **EHR variables** encode frailty, comorbidity burden, and surgical risk in a standardized form.
- **Fusing both modalities** provides an approximately 8 to 9 percent additional lift in AUC ROC compared with clinical variables alone.

---

## System Architecture

```
                       Data Generation Layer
   Synthetic EHR cohort (4,200 geriatric episodes) plus 7-day traces
        (Extensible to real clinical data; see EXTENDING.md)
                              |
              ------------------------------
              |                            |
    Actigraphy Raw Traces (NPZ)     EHR Variables (Tabular CSV)
    7 days x 1,440 minutes          4,200 x 47 features
              |                            |
              |   Statistical Encoder      |
              |   (Sleep efficiency,       |
              |    rest-activity rhythm    |
              |    amplitude,              |
              |    fragmentation)          |
              |                            |
              ------------------------------
                              |
                    Feature Fusion
                    (Clinical plus Actigraphy)
                    Engineered feature set
                              |
                    LightGBM Classifier
                    5-fold CV ensemble
                    Trained model artifact
                              |
              --------------------------------
              |               |               |
        Evaluation       Fairness         REST API
        Metrics          Audit            (FastAPI)
        (AUC, P@)        (Subgroups)       /predict
                                                |
                                       Streamlit Dashboard
                                       (Pre-op huddle)
```

---

## Key Features

### Multimodal Feature Fusion

- **EHR features** (47 clinical variables): age, comorbidity burden (the Charlson index), frailty, renal function, anesthesia risk, surgery type, and pre operative vitals.
- **Actigraphy features** (4 engineered embeddings):
  - Sleep efficiency, a 7 day mean
  - Rest activity rhythm amplitude, a measure of circadian strength
  - A fragmentation index, capturing sleep wake regularity
  - Pre operative sleep timing variance

### A Transparent Model Architecture

- LightGBM gradient boosting with 5 fold cross validation.
- SHAP integration for both local and global feature importance.
- Ablation analysis that quantifies the actigraphy contribution, an 8 to 9 percent lift in AUC.

### Fairness and Bias Auditing

Automated subgroup performance analysis, covering:

- Age stratification (65 to 75, 75 to 85, and 85 and older)
- Sex based parity (male and female)
- Frailty terciles (low, medium, and high)
- Metrics: AUC ROC, selection rate parity, and precision at the top 20 percent

### Production Ready Inference

- A FastAPI REST service, with health checks and batch prediction.
- A Streamlit pre operative dashboard, built for clinical huddle use.
- Sub second latency for real time decisions.
- Model versioning through joblib artifacts.

### Comprehensive Evaluation

- Cross validated AUC ROC: 0.70 typical, ranging from 0.67 to 0.72 across folds.
- Precision at the top 20 percent risk threshold: 0.50.
- Postoperative delirium incidence in the synthetic cohort: 26.2 percent.

---

## Performance Metrics

### Model Performance

| Metric | Value | Notes |
|---|---|---|
| AUC ROC (5 fold cross validation) | 0.700 (0.67 to 0.72) | Mean across folds, with the typical range shown; consistent with synthetic data |
| Precision at the top 20 percent | 0.50 | Of the 20 percent of patients flagged as highest risk, 50 percent go on to develop postoperative delirium |
| Recall at the top 20 percent | 0.38 | Captures 38 percent of all postoperative delirium cases within the top risk quintile |
| Cohort size | 4,200 | Simulated geriatric surgical episodes |
| Postoperative delirium incidence | 26.2 percent | Realistic for a high risk elderly population |

### Feature Contribution

- Actigraphy features contribute approximately 8 to 9 percent additional AUC lift over clinical variables alone.
- The top predictors are age, the Charlson comorbidity index, sleep fragmentation, and surgery duration.
- See `ablation_report.json` and the SHAP outputs for detailed attribution.

### Fairness Audit Results

See `fairness_report.json` after running `python fairness/audit.py`:

- **AUC parity:** well balanced across age and sex strata.
- **Fairness gaps:** minor AUC variance across frailty terciles, flagged for further investigation.
- **Recommendation:** monitor subgroup performance closely during pilot deployment.

---

## Installation and Quick Start

### Prerequisites

- Python 3.9 or later
- pip or conda
- Approximately 2 GB of disk space for synthetic data and model artifacts
- A Unix like environment (Linux or macOS) is recommended; Windows requires Git Bash or WSL

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

**Verify the installation:**

```bash
python -c "import lightgbm, fastapi, streamlit; print('All dependencies installed successfully')"
```

### The Quickstart Pipeline

Run the complete end to end workflow in approximately 2 minutes:

```bash
# 1. Generate synthetic data
python data/generate_data.py

# 2. Train the LightGBM model (produces the model artifact pod_lgbm.joblib)
python src/train.py

# 3. Evaluate model performance (produces metrics.json and ablation_report.json)
python src/evaluate.py

# 4. Audit fairness across demographic subgroups
python fairness/audit.py

# 5. Launch the FastAPI service (default: http://localhost:8000)
uvicorn serving.api:app --reload

# 6. In a separate terminal, launch the Streamlit dashboard (default: http://localhost:8501)
streamlit run serving/app.py
```

All outputs, including the model, data, and reports, are written to the repository root or its subdirectories.

---

## Repository Structure

```
POST_OPERATIVE_DELIRIUM_RISK_AI/
│
├── data/
│   ├── generate_data.py                 # Synthetic EHR plus actigraphy generation
│   ├── pod_episodes.csv                 # Tabular dataset (4,200 rows x 47 features plus label)
│   └── actigraphy_sample_traces.npz     # Raw minute-level traces (4,200 x 10,080)
│
├── src/
│   ├── actigraphy_encoder.py            # Statistical actigraphy feature extraction
│   ├── train.py                         # LightGBM model training plus cross-validation
│   └── evaluate.py                      # Performance evaluation plus ablation analysis
│
├── fairness/
│   └── audit.py                         # Fairness audit: subgroup AUC and parity metrics
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

1. **[MODEL_CARD.md](MODEL_CARD.md):** the formal model documentation, covering intended use and limitations, training data and cohort statistics, performance metrics and benchmarks, and fairness audit notes.

2. **[EXTENDING.md](EXTENDING.md):** the production roadmap, covering integration of real wearable actigraphy through GGIR, swapping in a true LSTM sequence encoder, the IRB and governance pathway for real clinical data, and deployment considerations such as containerization and HL7/FHIR integration.

3. **Code documentation:** inline docstrings are provided in all modules. Run `python -m pydoc src.train` for function signatures, or see the function headers directly in `src/train.py`, `src/actigraphy_encoder.py`, and related files.

---

## Usage Guide

### 1. Data Generation

Generate the synthetic EHR and actigraphy cohort:

```bash
python data/generate_data.py
```

**Output:**

- `pod_episodes.csv`: 4,200 rows, with 47 clinical features plus the postoperative delirium label.
- `actigraphy_sample_traces.npz`: raw minute level activity counts.

**To customize:** edit `data/generate_data.py` to modify the cohort size (4,200 by default), the postoperative delirium incidence (26.2 percent by default), and the actigraphy trace length (7 days at 1,440 minutes per day by default).

### 2. Model Training

Train the LightGBM classifier on the fused feature set:

```bash
python src/train.py
```

**Output:**

- `pod_lgbm.joblib`: the serialized model, ready for inference.
- Cross validation fold predictions, logged to standard output.

**Key parameters**, editable in `src/train.py`:

- `num_leaves`: LightGBM tree complexity (31 by default).
- `learning_rate`: the gradient boosting step size (0.05 by default).
- `n_estimators`: the number of boosting rounds (500 by default).

### 3. Evaluation and Metrics

Compute performance metrics and the ablation analysis:

```bash
python src/evaluate.py
```

**Output:**

- `metrics.json`: AUC ROC, precision at the top 20 percent, and decision thresholds.
- `ablation_report.json`: a comparison of the full model against a clinical only model.
- Detailed logs, written to standard output.

### 4. Fairness Auditing

Run the subgroup parity analysis:

```bash
python fairness/audit.py
```

**Output:**

- `fairness_report.json`: AUC, selection rates, and parity metrics broken down by age, sex, and frailty.
- A printout highlighting any fairness gaps found.

**Interpretation:**

- An AUC difference of less than 5 percent across strata is considered acceptable.
- Selection rate parity should be checked to ensure the top 20 percent selection is evenly distributed.
- Frailty gaps warrant further investigation before real world deployment.

### 5. Real Time Inference

#### Option A: The FastAPI Service

Start the server:

```bash
uvicorn serving.api:app --reload
```

**Endpoints:**

**`POST /predict`**, for batch prediction:

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

**`GET /health`**, for service status.

**Full API documentation:** http://localhost:8000/docs (a Swagger UI).

#### Option B: The Streamlit Dashboard

Launch the interactive dashboard:

```bash
streamlit run serving/app.py
```

**Features:**

- Patient lookup by ID
- Risk visualization, including a risk gauge and contributing risk factors
- Cohort level statistics
- A model performance overview
- Batch patient upload

See [Dashboard Preview](#dashboard-preview) below.

### 6. Exploratory Analysis

An interactive Jupyter notebook:

```bash
jupyter notebook notebooks/eda.ipynb
```

This includes cohort demographics, feature distributions, postoperative delirium incidence by risk strata, and actigraphy patterns among high risk patients.

---

## Dashboard Preview

![Postoperative Delirium Risk Dashboard](POST%20OPERATIVE%20DASHBOARD.png)

**Dashboard features:**

- **Risk gauge:** a patient level postoperative delirium risk score, on a 0 to 100 scale.
- **High risk alerts:** automatic flagging for scores above 61.4 percent, the top 20 percent threshold.
- **Contributing factors:** a SHAP based breakdown of the top predictive features.
- **Cohort overview:** aggregate statistics and distribution plots.
- **Model performance:** AUC ROC curves and calibration plots.
- **Comparison cohorts:** stratified performance by age, sex, and frailty.

---

## Model Architecture and Fairness

### Model Flow

```
                    Raw Actigraphy Traces
                    (7 days x 1,440 minutes x 4,200 patients)
                              |
                    Actigraphy Encoder
                    - Sleep efficiency
                    - Rest-activity rhythm amplitude
                    - Fragmentation index
                    - Sleep timing variance
                              |
              --------------------------------
              |                              |
     Actigraphy Features (4)          EHR Variables (43 clinical)
              |                              |
              --------------------------------
                              |
                    Feature Engineering
                    (Scaling, interactions)
                              |
                    LightGBM Classifier
                    - 5-fold cross validation
                    - 500 boosting rounds
                    - Max depth: 10, num_leaves: 31
                              |
              --------------------------------
              |                              |
     POD Risk Probability            Feature Importance
     (0 to 1)                        (SHAP values)
```

### Fairness Strategy

**A three tier audit:**

1. **Stratification variables:** age band (3 levels), sex (2 levels), and frailty (3 levels), producing 18 subgroups.
2. **Metrics:** AUC ROC, precision at the top 20 percent, and selection rate parity.
3. **Action thresholds:**
   - An AUC difference above 10 percent triggers investigation and documentation.
   - A selection rate imbalance above 15 percent triggers retraining with fairness constraints.
   - Consistent gaps by frailty trigger risk adjustment before deployment.

**Current status, on the synthetic cohort:**

- **Age and sex parity:** balanced, with an AUC difference below 5 percent.
- **Frailty gaps:** minor disparities have been flagged; real data validation is recommended before deployment.

---

## Extension and Deployment

### Near Term Enhancements (see EXTENDING.md)

1. **Real actigraphy integration.**
   - Replace synthetic traces with GGIR processed wearable data.
   - Validate on real pre operative patients (IRB approval required).
   - Expected AUC improvement: from 0.70 to a range of 0.75 to 0.80.

2. **An LSTM sequence encoder.**
   - Replace statistical features with a PyTorch LSTM operating on the 7 day traces.
   - Train end to end with a LightGBM head.
   - An additional AUC lift of approximately 3 to 5 percent is expected.

3. **Production deployment.**
   - Docker containerization of `api.py`.
   - HL7 and FHIR integration with EHR order entry.
   - Automated fairness auditing built into the CI/CD pipeline.
   - Model governance, including version control and an A/B testing framework.

4. **Governance and ethics.**
   - IRB approval for use of real patient data.
   - Informed consent and de identification protocols.
   - A bias monitoring dashboard, providing continuous fairness tracking.
   - Outcome tracking, following postoperative delirium incidence after implementation.

---

## Testing

Run the test suite:

```bash
pytest tests/ -v
```

**Coverage includes:**

- Data generation and shape validation
- Actigraphy encoder outputs (4 features per episode)
- Model training and cross validation
- API endpoint functionality
- Fairness audit subgroup stratification

---

## Requirements and Dependencies

| Package | Version | Purpose |
|---|---|---|
| numpy | 1.26 or later | Numerical computing and array handling |
| pandas | 2.0 or later | Data manipulation and analysis |
| scikit-learn | 1.3 or later | Preprocessing and metrics |
| lightgbm | 4.0 or later | The gradient boosting classifier |
| shap | 0.44 or later | Feature importance and explainability |
| joblib | 1.3 or later | Model serialization |
| fastapi | 0.110 or later | The REST API framework |
| uvicorn | 0.29 or later | The ASGI server for FastAPI |
| streamlit | optional | The interactive dashboard |
| pytest | 8.0 or later | Unit testing |

Full list: [requirements.txt](requirements.txt)

---

## Contributing

Contributions are welcome. Please follow these steps:

1. **Fork** this repository.
2. **Create a feature branch** (`git checkout -b feature/your-feature`).
3. **Commit** your changes with clear messages.
4. **Push** to the branch and **open a pull request**.

**Guidelines:**

- Follow PEP 8 style, using `black` for formatting.
- Add docstrings to all functions.
- Include unit tests for new modules.
- Update this README if adding major features.

---

## Citation

If this work is used in research or clinical settings, please cite it as follows:

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

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

**Summary:** free use for research, education, and commercial purposes with attribution. No warranty is implied.

---

## Acknowledgments

### Scientific Foundation

- The HELP (Hospital Elder Life Program) prevention framework.
- Recent literature on actigraphy and delirium risk.
- Best practices for fairness in clinical AI, drawing on FDA and WHO guidelines.

### Technical Stack

- The LightGBM team, for a robust gradient boosting implementation.
- The Plotly and Streamlit communities, for visualization tooling.
- The SHAP authors, for the interpretability framework.
- The FastAPI team, for a modern Python API framework.

### Contributors

- Anthony Chinedu Echem (primary author)

---

## Support and Contact

### Troubleshooting

**Model training is slow.**
Reduce `n_estimators` in `src/train.py`, or reduce the cohort size in `data/generate_data.py`.

**"ImportError: No module named 'lightgbm'"**
Run `pip install lightgbm`, or re run `pip install -r requirements.txt`.

**The FastAPI service will not start.**
Ensure port 8000 is available, or use `uvicorn serving.api:app --port 8001` to specify a different port.

**The fairness report shows large AUC gaps.**
This is expected on synthetic data; validate against a real clinical cohort (see EXTENDING.md).

### Getting Help

- Check existing [GitHub Issues](https://github.com/ANTHONY-CHINEDU-ECHEM/POST_OPERATIVE_DELIRIUM_RISK_AI/issues).
- Review [EXTENDING.md](EXTENDING.md) for deployment questions.
- See [MODEL_CARD.md](MODEL_CARD.md) for model specific details.

---

## Disclaimer

This is a reference implementation for research and educational purposes. Clinical deployment requires:

- Institutional Review Board (IRB) approval
- Clinical validation on real patient cohorts
- Regulatory compliance, including FDA 510(k) clearance if applicable
- Clinical governance and outcome tracking

This system is not a substitute for clinical judgment. Model outputs are decision support tools only.

---

<div align="center">

**Last updated:** 2024
**Status:** Active development
**Questions?** Open a GitHub Issue or check [EXTENDING.md](EXTENDING.md)

</div>
