"""
models/model_selector.py
Compare all model metrics per state and persist the best model name.
"""

import pandas as pd
import json
import os
import logging

logger = logging.getLogger(__name__)
MODEL_DIR = "models/saved"
BEST_MODELS_PATH = os.path.join(MODEL_DIR, "best_models.json")


def select_best_models(
    sarima_df: pd.DataFrame,
    prophet_df: pd.DataFrame,
    xgb_df: pd.DataFrame,
    lstm_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each state, pick the model with the lowest RMSE.
    Returns a DataFrame: state, best_model, RMSE, MAE, MAPE%
    """
    all_dfs = []
    for df, name in [(sarima_df, "SARIMA"), (prophet_df, "Prophet"),
                     (xgb_df, "XGBoost"), (lstm_df, "LSTM")]:
        if df is not None and not df.empty:
            tmp = df.copy()
            tmp["model"] = name
            all_dfs.append(tmp)

    combined = pd.concat(all_dfs, ignore_index=True)
    combined["RMSE"] = pd.to_numeric(combined["RMSE"], errors="coerce")

    best = combined.loc[combined.groupby("state")["RMSE"].idxmin()].reset_index(drop=True)
    best = best.rename(columns={"model": "best_model"})

    os.makedirs(MODEL_DIR, exist_ok=True)
    best_dict = dict(zip(best["state"], best["best_model"]))
    with open(BEST_MODELS_PATH, "w") as f:
        json.dump(best_dict, f, indent=2)

    logger.info(f"Best model counts:\n{best['best_model'].value_counts()}")
    return best


def get_best_model(state: str) -> str:
    with open(BEST_MODELS_PATH) as f:
        best_dict = json.load(f)
    return best_dict.get(state, "Prophet")   # default to Prophet


def load_all_metrics() -> pd.DataFrame:
    """Read saved best_models.json as a DataFrame."""
    with open(BEST_MODELS_PATH) as f:
        best_dict = json.load(f)
    return pd.DataFrame(list(best_dict.items()), columns=["state", "best_model"])
