"""Lightweight statistical encoder for raw minute-level actigraphy traces.

Stands in for the LSTM encoder described in the portfolio briefing:
compresses each 7-day trace into a small set of circadian-rhythm summary
features (the same signal family GGIR extracts: amplitude, inter-daily
stability, intra-daily variability, and a fragmentation proxy).
"""
import numpy as np

MIN_PER_DAY = 1440
DAYS = 7


def encode(trace: np.ndarray) -> dict:
    hour = (np.arange(len(trace)) / 60.0) % 24
    night = (hour < 6) | (hour > 22)
    day = ~night

    day_mean = trace[day].mean()
    night_mean = trace[night].mean()
    amplitude = day_mean - night_mean  # rest-activity rhythm amplitude

    # Inter-daily stability: how similar the daily pattern is day-to-day
    reshaped = trace[: DAYS * MIN_PER_DAY].reshape(DAYS, MIN_PER_DAY)
    hourly = reshaped.reshape(DAYS, 24, 60).mean(axis=2)
    ids = float(np.var(hourly.mean(axis=0)) / (np.var(hourly) + 1e-6))

    # Intra-daily variability: hour-to-hour fragmentation within a day
    diffs = np.diff(hourly, axis=1)
    iv = float(np.mean(diffs ** 2) / (np.var(hourly) + 1e-6))

    sleep_efficiency = 1.0 - (trace[night] > np.percentile(trace, 60)).mean()

    return {
        "actg_amplitude": float(amplitude),
        "actg_interdaily_stability": ids,
        "actg_intradaily_variability": iv,
        "actg_sleep_efficiency": float(sleep_efficiency),
        "actg_night_mean_activity": float(night_mean),
    }


def encode_batch(traces: np.ndarray) -> "pd.DataFrame":
    import pandas as pd
    rows = [encode(t) for t in traces]
    return pd.DataFrame(rows)
