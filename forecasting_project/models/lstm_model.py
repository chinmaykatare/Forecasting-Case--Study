"""
models/lstm_model.py
Train LSTM per state using a sliding window approach.
"""

import pandas as pd
import numpy as np
import joblib
import os
import logging
import json

logger = logging.getLogger(__name__)
MODEL_DIR = "models/saved"
os.makedirs(MODEL_DIR, exist_ok=True)

LOOKBACK = 12   # 12-week input window


def build_sequences(series: np.ndarray, lookback: int = LOOKBACK):
    X, y = [], []
    for i in range(lookback, len(series)):
        X.append(series[i - lookback: i])
        y.append(series[i])
    return np.array(X), np.array(y)


def train_lstm(train_df: pd.DataFrame, val_df: pd.DataFrame, state: str):
    # Import TF here so the module loads even without TF installed
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    from sklearn.preprocessing import MinMaxScaler
    from utils.metrics import evaluate

    tr = train_df[train_df["state"] == state].sort_values("date")["sales"].values.astype(float)
    va = val_df[val_df["state"] == state].sort_values("date")["sales"].values.astype(float)

    # Scale
    scaler = MinMaxScaler()
    tr_scaled = scaler.fit_transform(tr.reshape(-1, 1)).flatten()
    va_scaled = scaler.transform(va.reshape(-1, 1)).flatten()

    # Sequences
    full_scaled = np.concatenate([tr_scaled, va_scaled])
    X_tr, y_tr = build_sequences(tr_scaled)
    # For val, seed from end of train
    X_all, y_all = build_sequences(full_scaled)
    X_va = X_all[len(X_tr):]
    y_va_scaled = y_all[len(X_tr):]

    X_tr = X_tr.reshape(-1, LOOKBACK, 1)
    X_va = X_va.reshape(-1, LOOKBACK, 1)

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(LOOKBACK, 1)),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    es = EarlyStopping(patience=10, restore_best_weights=True)
    model.fit(X_tr, y_tr, epochs=100, batch_size=16,
              validation_data=(X_va, y_va_scaled),
              callbacks=[es], verbose=0)

    preds_scaled = model.predict(X_va, verbose=0).flatten()
    preds = scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
    preds = np.clip(preds, 0, None)
    metrics = evaluate(va[:len(preds)], preds, "LSTM")

    # Save model + scaler
    model.save(os.path.join(MODEL_DIR, f"lstm_{state.replace(' ', '_')}.keras"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, f"lstm_scaler_{state.replace(' ', '_')}.pkl"))

    # Save tail of scaled series for future forecasting
    tail = full_scaled[-LOOKBACK:].tolist()
    with open(os.path.join(MODEL_DIR, f"lstm_tail_{state.replace(' ', '_')}.json"), "w") as f:
        json.dump(tail, f)

    logger.info(f"  LSTM {state}  |  RMSE={metrics['RMSE']}")
    return metrics, preds


def train_all_lstm(train_df, val_df):
    results = []
    for state in train_df["state"].unique():
        try:
            m, _ = train_lstm(train_df, val_df, state)
            m["state"] = state
            results.append(m)
        except Exception as e:
            logger.warning(f"  LSTM failed for {state}: {e}")
    return pd.DataFrame(results)


def forecast_lstm(state: str, n_periods: int = 8):
    import tensorflow as tf
    import json

    safe = state.replace(" ", "_")
    model  = tf.keras.models.load_model(os.path.join(MODEL_DIR, f"lstm_{safe}.keras"))
    scaler = joblib.load(os.path.join(MODEL_DIR, f"lstm_scaler_{safe}.pkl"))
    with open(os.path.join(MODEL_DIR, f"lstm_tail_{safe}.json")) as f:
        tail = json.load(f)

    window = np.array(tail, dtype=float)
    preds  = []
    for _ in range(n_periods):
        x    = window[-LOOKBACK:].reshape(1, LOOKBACK, 1)
        pred_scaled = model.predict(x, verbose=0)[0][0]
        pred = float(scaler.inverse_transform([[pred_scaled]])[0][0])
        pred = max(pred, 0)
        preds.append(pred)
        window = np.append(window, pred_scaled)

    return preds
