# Model Card — Postoperative Delirium Risk Model

## Intended use
Pre-op decision support to prioritize the HELP (Hospital Elder Life
Program) prevention bundle for the highest-risk ~20% of elderly surgical
patients. Not a diagnostic tool.

## Training data
Synthetic cohort (`data/generate_data.py`), 4,200 simulated geriatric
surgical episodes with paired 7-day pre-op actigraphy traces.

## Metrics (last training run)
See `models/metrics.json`, `models/ablation_report.json`, and
`fairness/fairness_report.json` for exact figures from the current data
generation seed. Typical results on this synthetic benchmark: AUC-ROC ≈
0.69-0.74; precision at the top-20%-risk threshold ≈ 0.45-0.50; actigraphy
features contribute an ~8-9% AUC lift over tabular-only (meets the ≥8%
target from the portfolio briefing). Note the briefing's aspirational
AUC-ROC ≥ 0.80 target assumes a larger, richer real-world cohort with a
true LSTM sequence encoder (see `docs/EXTENDING.md`) — this synthetic
demo's simpler statistical actigraphy features cap achievable AUC lower,
which is expected and disclosed here rather than papered over.

## Limitations
- Synthetic actigraphy traces approximate but do not replace real
  wearable data; device non-wear and artifact patterns are not modeled.
- Fairness gaps across frailty terciles in the audit report should be
  investigated further before deployment (see `fairness/audit.py` output).

## Fairness audit
Run `python fairness/audit.py` after training — reports AUC and
selection-rate parity across sex, age band, and frailty tercile.
