"""
models/prophet_model.py
Train Facebook Prophet per state.
"""

import pandas as pd
import numpy as np
import joblib
import os
import logging
from prophet import Prophet
from utils.metrics import evaluate

logger = logging.getLogger(__name__)
MODEL_DIR = "models/saved"
os.makedirs(MODEL_DIR, exist_ok=True)


def train_prophet(train_df: pd.DataFrame, val_df: pd.DataFrame, state: str):
    tr = train_df[train_df["state"] == state].sort_values("date")[["date", "sales"]].copy()
    va = val_df[val_df["state"] == state].sort_values("date")
    tr = tr.rename(columns={"date": "ds", "sales": "y"})

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
        interval_width=0.80,
    )
    model.fit(tr)

    future = model.make_future_dataframe(periods=len(va), freq="W")
    forecast = model.predict(future)
    preds = forecast["yhat"].values[-len(va):]
    preds = np.clip(preds, 0, None)
    metrics = evaluate(va["sales"].values, preds, "Prophet")

    path = os.path.join(MODEL_DIR, f"prophet_{state.replace(' ', '_')}.pkl")
    joblib.dump(model, path)
    logger.info(f"  Prophet {state}  |  RMSE={metrics['RMSE']}")
    return metrics, preds


def train_all_prophet(train_df, val_df):
    results = []
    for state in train_df["state"].unique():
        try:
            m, _ = train_prophet(train_df, val_df, state)
            m["state"] = state
            results.append(m)
        except Exception as e:
            logger.warning(f"  Prophet failed for {state}: {e}")
    return pd.DataFrame(results)


def forecast_prophet(state: str, last_date: str, n_periods: int = 8):
    path = os.path.join(MODEL_DIR, f"prophet_{state.replace(' ', '_')}.pkl")
    model = joblib.load(path)
    last = pd.to_datetime(last_date)
    future = pd.DataFrame({"ds": pd.date_range(last + pd.Timedelta(weeks=1), periods=n_periods, freq="W")})
    forecast = model.predict(future)
    return np.clip(forecast["yhat"].values, 0, None).tolist()
