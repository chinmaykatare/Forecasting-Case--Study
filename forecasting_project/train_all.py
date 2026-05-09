"""
train_all.py
============
Master training script.
Usage:
    python train_all.py --data data/sales_data.xlsx

Trains SARIMA, Prophet, XGBoost, LSTM for every state,
selects the best model per state, and saves everything to models/saved/.
"""

import argparse
import os
import sys
import logging
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Make sure utils & models are importable
sys.path.insert(0, os.path.dirname(__file__))

from utils.data_loader import full_pipeline, train_val_split
from models.sarima_model import train_all_sarima
from models.prophet_model import train_all_prophet
from models.xgboost_model import train_all_xgb
from models.lstm_model import train_all_lstm
from models.model_selector import select_best_models


def main(data_path: str, skip_sarima: bool = False, skip_lstm: bool = False):
    # ── 1. Load & engineer features ─────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 1: Loading and preprocessing data …")
    df = full_pipeline(data_path)
    df.to_csv("data/processed_data.csv", index=False)
    logger.info(f"Processed data saved → data/processed_data.csv  ({len(df):,} rows)")

    # ── 2. Train / val split ─────────────────────────────────────────────
    train_df, val_df = train_val_split(df, val_weeks=8)
    logger.info(f"Train rows: {len(train_df):,}  |  Val rows: {len(val_df):,}")

    # ── 3. SARIMA ────────────────────────────────────────────────────────
    sarima_results = None
    if not skip_sarima:
        logger.info("=" * 60)
        logger.info("STEP 2: Training SARIMA …  (this can take 20-40 min)")
        # Use raw (non-feature-engineered) data for SARIMA
        raw_df = df[["date", "state", "sales"]].copy()
        raw_train, raw_val = train_val_split(raw_df, val_weeks=8)
        sarima_results = train_all_sarima(raw_train, raw_val)
        sarima_results.to_csv("outputs/sarima_metrics.csv", index=False)
        logger.info("SARIMA done. Metrics → outputs/sarima_metrics.csv")
    else:
        logger.info("SARIMA skipped (--skip_sarima flag)")

    # ── 4. Prophet ───────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 3: Training Prophet …")
    raw_df = df[["date", "state", "sales"]].copy()
    raw_train, raw_val = train_val_split(raw_df, val_weeks=8)
    prophet_results = train_all_prophet(raw_train, raw_val)
    prophet_results.to_csv("outputs/prophet_metrics.csv", index=False)
    logger.info("Prophet done. Metrics → outputs/prophet_metrics.csv")

    # ── 5. XGBoost ───────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 4: Training XGBoost …")
    xgb_results = train_all_xgb(train_df, val_df)
    xgb_results.to_csv("outputs/xgb_metrics.csv", index=False)
    logger.info("XGBoost done. Metrics → outputs/xgb_metrics.csv")

    # ── 6. LSTM ──────────────────────────────────────────────────────────
    lstm_results = None
    if not skip_lstm:
        logger.info("=" * 60)
        logger.info("STEP 5: Training LSTM …  (GPU recommended)")
        lstm_results = train_all_lstm(train_df, val_df)
        lstm_results.to_csv("outputs/lstm_metrics.csv", index=False)
        logger.info("LSTM done. Metrics → outputs/lstm_metrics.csv")
    else:
        logger.info("LSTM skipped (--skip_lstm flag)")

    # ── 7. Select best model ─────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 6: Selecting best model per state …")
    best = select_best_models(sarima_results, prophet_results, xgb_results, lstm_results)
    best.to_csv("outputs/best_models.csv", index=False)
    logger.info("Best models saved → outputs/best_models.csv")
    logger.info(best[["state", "best_model", "RMSE"]].to_string(index=False))

    logger.info("=" * 60)
    logger.info("✅  Training complete! Now run: uvicorn api.main:app --reload")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",         default="data/sales_data.xlsx", help="Path to Excel/CSV")
    parser.add_argument("--skip_sarima",  action="store_true", help="Skip SARIMA (fast run)")
    parser.add_argument("--skip_lstm",    action="store_true", help="Skip LSTM (fast run)")
    args = parser.parse_args()

    os.makedirs("outputs", exist_ok=True)
    os.makedirs("models/saved", exist_ok=True)

    main(args.data, skip_sarima=args.skip_sarima, skip_lstm=args.skip_lstm)
