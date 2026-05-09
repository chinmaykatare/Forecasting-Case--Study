"""
models/sarima_model.py
Train SARIMA per state, save with joblib.
"""

import pandas as pd
import numpy as np
import joblib
import os
import logging
from pmdarima import auto_arima
from utils.metrics import evaluate

logger = logging.getLogger(__name__)
MODEL_DIR = "models/saved"
os.makedirs(MODEL_DIR, exist_ok=True)


def train_sarima(train_df: pd.DataFrame, val_df: pd.DataFrame, state: str):
    """Fit auto_arima for one state, return val predictions + metrics."""
    series = train_df[train_df["state"] == state].sort_values("date")["sales"].values
    val_series = val_df[val_df["state"] == state].sort_values("date")["sales"].values
    n_forecast = len(val_series)

    logger.info(f"  SARIMA fitting {state} (n={len(series)}) …")
    model = auto_arima(
        series,
        seasonal=True, m=52,          # weekly seasonality
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore",
        max_p=3, max_q=3, max_P=2, max_Q=2,
        information_criterion="aic",
        n_jobs=1,
    )

    preds = model.predict(n_periods=n_forecast)
    preds = np.clip(preds, 0, None)
    metrics = evaluate(val_series, preds, "SARIMA")

    path = os.path.join(MODEL_DIR, f"sarima_{state.replace(' ', '_')}.pkl")
    joblib.dump(model, path)
    logger.info(f"  Saved {path}  |  RMSE={metrics['RMSE']}")
    return metrics, preds


def train_all_sarima(train_df, val_df):
    results = []
    states  = train_df["state"].unique()
    for state in states:
        try:
            m, _ = train_sarima(train_df, val_df, state)
            m["state"] = state
            results.append(m)
        except Exception as e:
            logger.warning(f"  SARIMA failed for {state}: {e}")
    return pd.DataFrame(results)


def forecast_sarima(state: str, n_periods: int = 8):
    path = os.path.join(MODEL_DIR, f"sarima_{state.replace(' ', '_')}.pkl")
    model = joblib.load(path)
    preds = model.predict(n_periods=n_periods)
    return np.clip(preds, 0, None).tolist()
