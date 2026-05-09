"""
models/xgboost_model.py
Train XGBoost with lag + calendar features per state.
"""

import pandas as pd
import numpy as np
import joblib
import os
import logging
from xgboost import XGBRegressor
from utils.metrics import evaluate

logger = logging.getLogger(__name__)
MODEL_DIR = "models/saved"
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURE_COLS = [
    "lag_1w", "lag_2w", "lag_4w",
    "roll_mean_4w", "roll_std_4w",
    "roll_mean_8w", "roll_std_8w",
    "week_of_year", "month", "quarter",
    "day_of_week", "is_holiday",
]


def train_xgb(train_df: pd.DataFrame, val_df: pd.DataFrame, state: str):
    tr = train_df[train_df["state"] == state].sort_values("date").dropna(subset=FEATURE_COLS)
    va = val_df[val_df["state"] == state].sort_values("date").dropna(subset=FEATURE_COLS)

    X_tr, y_tr = tr[FEATURE_COLS], tr["sales"]
    X_va, y_va = va[FEATURE_COLS], va["sales"]

    model = XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

    preds = np.clip(model.predict(X_va), 0, None)
    metrics = evaluate(y_va.values, preds, "XGBoost")

    path = os.path.join(MODEL_DIR, f"xgb_{state.replace(' ', '_')}.pkl")
    joblib.dump(model, path)
    logger.info(f"  XGBoost {state}  |  RMSE={metrics['RMSE']}")
    return metrics, preds


def train_all_xgb(train_df, val_df):
    results = []
    for state in train_df["state"].unique():
        try:
            m, _ = train_xgb(train_df, val_df, state)
            m["state"] = state
            results.append(m)
        except Exception as e:
            logger.warning(f"  XGBoost failed for {state}: {e}")
    return pd.DataFrame(results)


def forecast_xgb(state: str, last_known_row: dict, n_periods: int = 8):
    """
    Recursive multi-step forecast using last known feature row as seed.
    """
    path = os.path.join(MODEL_DIR, f"xgb_{state.replace(' ', '_')}.pkl")
    model = joblib.load(path)

    window = list(last_known_row.get("history", []))   # recent sales history
    preds  = []
    row    = last_known_row.copy()

    for i in range(n_periods):
        X = pd.DataFrame([{c: row.get(c, 0) for c in FEATURE_COLS}])
        pred = float(np.clip(model.predict(X)[0], 0, None))
        preds.append(pred)
        window.append(pred)

        # Shift lag features
        row["lag_4w"] = window[-4] if len(window) >= 4 else pred
        row["lag_2w"] = window[-2] if len(window) >= 2 else pred
        row["lag_1w"] = window[-1]
        row["roll_mean_4w"] = np.mean(window[-4:]) if len(window) >= 4 else pred
        row["roll_std_4w"]  = np.std(window[-4:])  if len(window) >= 4 else 0
        row["roll_mean_8w"] = np.mean(window[-8:]) if len(window) >= 8 else pred
        row["roll_std_8w"]  = np.std(window[-8:])  if len(window) >= 8 else 0
        row["week_of_year"] = (row.get("week_of_year", 1) % 52) + 1
        row["month"]        = ((row.get("month", 1)) % 12) + 1

    return preds
