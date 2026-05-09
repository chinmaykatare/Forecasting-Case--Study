"""
utils/data_loader.py
Loads, cleans, resamples the Excel dataset.
"""

import pandas as pd
import numpy as np
import holidays
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 1.  LOAD RAW DATA
# ─────────────────────────────────────────────
def load_raw_data(filepath: str) -> pd.DataFrame:
    """Read Excel / CSV file and return a raw DataFrame."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext in [".xlsx", ".xls"]:
        df = pd.read_excel(filepath)
    else:
        df = pd.read_csv(filepath)

    logger.info(f"Loaded {len(df):,} rows, {df.shape[1]} columns from {filepath}")
    logger.info(f"Columns: {df.columns.tolist()}")
    return df


# ─────────────────────────────────────────────
# 2.  STANDARDISE COLUMN NAMES
# ─────────────────────────────────────────────
def standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename columns to: date, state, sales.
    Handles multiple possible column naming conventions.
    """
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Map common variants
    rename_map = {}
    for col in df.columns:
        if col in ["week", "date", "week_date", "order_date", "transaction_date", "order_week"]:
            rename_map[col] = "date"
        elif col in ["state", "state_name", "region", "location"]:
            rename_map[col] = "state"
        elif col in ["sales", "revenue", "amount", "total_sales", "weekly_sales", "sale"]:
            rename_map[col] = "sales"

    df = df.rename(columns=rename_map)

    required = {"date", "state", "sales"}
    missing = required - set(df.columns)
    if missing:
        logger.warning(f"Could not auto-detect columns: {missing}. Columns present: {df.columns.tolist()}")
        # Fallback: assume first three columns are date, state, sales
        df.columns = ["date", "state", "sales"] + list(df.columns[3:])
        logger.info("Fell back to positional column assignment.")

    return df


# ─────────────────────────────────────────────
# 3.  PARSE & CLEAN
# ─────────────────────────────────────────────
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Parse date
    df["date"] = pd.to_datetime(df["date"], infer_datetime_format=True, errors="coerce")
    before = len(df)
    df = df.dropna(subset=["date"])
    logger.info(f"Dropped {before - len(df)} rows with unparseable dates.")

    # Clean sales
    df["sales"] = pd.to_numeric(df["sales"], errors="coerce")
    df["sales"] = df["sales"].clip(lower=0)          # no negative sales
    df["state"] = df["state"].str.strip().str.title()

    logger.info(f"Unique states: {df['state'].nunique()}")
    logger.info(f"Date range: {df['date'].min()} → {df['date'].max()}")
    return df


# ─────────────────────────────────────────────
# 4.  RESAMPLE TO STRICT WEEKLY (Sunday anchor)
# ─────────────────────────────────────────────
def resample_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each state, resample to a strict W-SUN frequency.
    Missing weeks are filled with forward-fill → backward-fill → 0.
    """
    df = df.copy()
    results = []

    for state, grp in df.groupby("state"):
        grp = grp.set_index("date").sort_index()
        # Keep only date + sales; aggregate duplicates
        grp = grp[["sales"]].resample("W").sum()

        # Fill gaps
        full_idx = pd.date_range(grp.index.min(), grp.index.max(), freq="W")
        grp = grp.reindex(full_idx)
        missing = grp["sales"].isna().sum()
        if missing:
            logger.info(f"  {state}: {missing} missing weeks → interpolating")
        grp["sales"] = grp["sales"].interpolate(method="linear").bfill().ffill().fillna(0)
        grp["state"] = state
        grp.index.name = "date"
        results.append(grp.reset_index())

    combined = pd.concat(results, ignore_index=True)
    logger.info(f"After resampling: {len(combined):,} rows, {combined['state'].nunique()} states")
    return combined


# ─────────────────────────────────────────────
# 5.  FEATURE ENGINEERING
# ─────────────────────────────────────────────
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values(["state", "date"]).reset_index(drop=True)

    # ── Calendar features ──────────────────────
    df["week_of_year"]  = df["date"].dt.isocalendar().week.astype(int)
    df["month"]         = df["date"].dt.month
    df["quarter"]       = df["date"].dt.quarter
    df["year"]          = df["date"].dt.year
    df["day_of_week"]   = df["date"].dt.dayofweek   # 0=Mon … 6=Sun

    # ── US holiday flag ────────────────────────
    us_holidays = holidays.US(years=range(2018, 2026))
    df["is_holiday"] = df["date"].dt.date.apply(lambda d: int(d in us_holidays))

    # ── Per-state lag & rolling features ──────
    lag_cols = []
    roll_cols = []

    for state, grp in df.groupby("state"):
        idx = grp.index
        s   = grp["sales"]

        # Lags: 1, 2, 4 weeks (equivalent to t-1, t-2, t-4 in weekly data)
        for lag in [1, 2, 4]:
            col = f"lag_{lag}w"
            df.loc[idx, col] = s.shift(lag).values
            lag_cols.append(col)

        # Rolling mean & std (4-week, 8-week)
        for window in [4, 8]:
            df.loc[idx, f"roll_mean_{window}w"] = s.shift(1).rolling(window).mean().values
            df.loc[idx, f"roll_std_{window}w"]  = s.shift(1).rolling(window).std().values
            roll_cols.append(f"roll_mean_{window}w")
            roll_cols.append(f"roll_std_{window}w")

    # Drop rows where lag features are NaN (initial rows per state)
    df = df.dropna(subset=lag_cols).reset_index(drop=True)
    logger.info(f"After feature engineering: {len(df):,} rows")
    return df


# ─────────────────────────────────────────────
# 6.  TRAIN / VALIDATION SPLIT  (no leakage)
# ─────────────────────────────────────────────
def train_val_split(df: pd.DataFrame, val_weeks: int = 8):
    """
    For each state split chronologically: last `val_weeks` rows = validation.
    Returns (train_df, val_df).
    """
    trains, vals = [], []
    for state, grp in df.groupby("state"):
        grp = grp.sort_values("date")
        trains.append(grp.iloc[:-val_weeks])
        vals.append(grp.iloc[-val_weeks:])
    return pd.concat(trains), pd.concat(vals)


# ─────────────────────────────────────────────
# 7.  FULL PIPELINE
# ─────────────────────────────────────────────
def full_pipeline(filepath: str) -> pd.DataFrame:
    df = load_raw_data(filepath)
    df = standardise_columns(df)
    df = clean_data(df)
    df = resample_weekly(df)
    df = add_features(df)
    return df
