"""
api/main.py
===========
FastAPI REST service for sales forecasting.

Endpoints:
  GET  /health                      → health check
  GET  /states                      → list all available states
  POST /forecast                    → forecast for one state
  POST /forecast/bulk               → forecast for multiple states
  GET  /models/best                 → best model per state
  GET  /metrics                     → model comparison metrics
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from typing import List, Optional

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models.model_selector import get_best_model
from utils.data_loader import full_pipeline

# ── Lazy imports (only if model file exists) ────────────────────────
def _forecast_sarima(state, n):
    from models.sarima_model import forecast_sarima
    return forecast_sarima(state, n)

def _forecast_prophet(state, last_date, n):
    from models.prophet_model import forecast_prophet
    return forecast_prophet(state, last_date, n)

def _forecast_xgb(state, seed_row, n):
    from models.xgboost_model import forecast_xgb
    return forecast_xgb(state, seed_row, n)

def _forecast_lstm(state, n):
    from models.lstm_model import forecast_lstm
    return forecast_lstm(state, n)

# ── Logger ───────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── App ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="Sales Forecasting API",
    description="End-to-end time-series forecasting: SARIMA, Prophet, XGBoost, LSTM",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# ── In-memory cache loaded at startup ───────────────────────────────
DATA_CACHE: dict = {}

@app.on_event("startup")
def load_data():
    data_path = os.environ.get("DATA_PATH", "data/processed_data.csv")
    if os.path.exists(data_path):
        df = pd.read_csv(data_path, parse_dates=["date"])
        DATA_CACHE["df"]     = df
        DATA_CACHE["states"] = sorted(df["state"].unique().tolist())
        logger.info(f"Loaded {len(df):,} rows | {len(DATA_CACHE['states'])} states")
    else:
        logger.warning(f"Data file not found at {data_path}. Run train_all.py first.")
        DATA_CACHE["df"]     = pd.DataFrame()
        DATA_CACHE["states"] = []


# ── Schemas ──────────────────────────────────────────────────────────
class ForecastRequest(BaseModel):
    state:    str
    n_weeks:  int = 8
    model:    Optional[str] = None    # None → use best model

class BulkForecastRequest(BaseModel):
    states:   List[str]
    n_weeks:  int = 8

class ForecastPoint(BaseModel):
    date:  str
    sales: float

class ForecastResponse(BaseModel):
    state:       str
    model_used:  str
    n_weeks:     int
    forecast:    List[ForecastPoint]
    generated_at: str


# ── Helper ───────────────────────────────────────────────────────────
def _run_forecast(state: str, n_weeks: int, model_override: Optional[str] = None):
    df   = DATA_CACHE.get("df", pd.DataFrame())
    sdf  = df[df["state"] == state].sort_values("date") if not df.empty else pd.DataFrame()

    if sdf.empty:
        raise HTTPException(404, f"No data for state: {state}")

    last_date = sdf["date"].iloc[-1]
    model_name = model_override or get_best_model(state)

    # Build future dates
    future_dates = [
        (last_date + timedelta(weeks=i+1)).strftime("%Y-%m-%d")
        for i in range(n_weeks)
    ]

    try:
        if model_name == "SARIMA":
            preds = _forecast_sarima(state, n_weeks)

        elif model_name == "Prophet":
            preds = _forecast_prophet(state, str(last_date.date()), n_weeks)

        elif model_name == "XGBoost":
            last_row = sdf.iloc[-1].to_dict()
            history  = sdf["sales"].tolist()
            last_row["history"] = history
            preds = _forecast_xgb(state, last_row, n_weeks)

        elif model_name == "LSTM":
            preds = _forecast_lstm(state, n_weeks)

        else:
            raise HTTPException(400, f"Unknown model: {model_name}")

    except FileNotFoundError:
        raise HTTPException(503, f"Model for {state} not found. Run train_all.py first.")

    preds = [max(0, float(p)) for p in preds]

    return ForecastResponse(
        state        = state,
        model_used   = model_name,
        n_weeks      = n_weeks,
        forecast     = [ForecastPoint(date=d, sales=round(p, 2))
                        for d, p in zip(future_dates, preds)],
        generated_at = datetime.utcnow().isoformat(),
    )


# ── Routes ───────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
def health():
    return {"status": "ok", "states_loaded": len(DATA_CACHE.get("states", []))}


@app.get("/states", tags=["Data"])
def list_states():
    return {"states": DATA_CACHE.get("states", [])}


@app.post("/forecast", response_model=ForecastResponse, tags=["Forecast"])
def forecast(req: ForecastRequest):
    if req.n_weeks < 1 or req.n_weeks > 52:
        raise HTTPException(400, "n_weeks must be between 1 and 52")
    return _run_forecast(req.state, req.n_weeks, req.model)


@app.post("/forecast/bulk", tags=["Forecast"])
def forecast_bulk(req: BulkForecastRequest):
    results = []
    errors  = []
    for state in req.states:
        try:
            results.append(_run_forecast(state, req.n_weeks).dict())
        except HTTPException as e:
            errors.append({"state": state, "error": e.detail})
    return {"results": results, "errors": errors}


@app.get("/models/best", tags=["Models"])
def best_models():
    path = "models/saved/best_models.json"
    if not os.path.exists(path):
        raise HTTPException(503, "Models not trained yet. Run train_all.py first.")
    with open(path) as f:
        return json.load(f)


@app.get("/metrics", tags=["Models"])
def model_metrics():
    out = {}
    for name, fname in [("XGBoost", "outputs/xgb_metrics.csv"),
                        ("Prophet", "outputs/prophet_metrics.csv"),
                        ("SARIMA",  "outputs/sarima_metrics.csv"),
                        ("LSTM",    "outputs/lstm_metrics.csv")]:
        if os.path.exists(fname):
            out[name] = pd.read_csv(fname).to_dict(orient="records")
    return out


@app.get("/", tags=["System"])
def root():
    return {
        "message": "Sales Forecasting API",
        "docs":    "/docs",
        "health":  "/health",
    }
