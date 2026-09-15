# Extending this reference implementation

1. **Real actigraphy.** Replace `data/generate_data.py`'s simulated traces
   with real GGIR-processed accelerometer output from a wearable pilot;
   keep the same `actigraphy_raw.npz` shape (episodes x minutes).
2. **True sequence model.** Swap `src/actigraphy_encoder.py`'s statistical
   summary for a PyTorch LSTM/1D-CNN encoder trained end-to-end with the
   LightGBM head (or replace the whole fusion model with a single deep
   multimodal network).
3. **IRB pathway.** Formalize the consent and de-identification process
   before any real wearable data collection — see the governance notes in
   the portfolio briefing deck.
4. **Deployment.** Containerize `serving/api.py`, wire into the EHR order
   set via HL7/FHIR, and connect `fairness/audit.py` to a scheduled CI job
   that reruns on every retrain.
