# 📊 End-to-End Time Series Forecasting System

**Assignment:** Data Science — Sales Forecasting with REST API  
**Models:** SARIMA · Prophet · XGBoost · LSTM  
**Framework:** FastAPI  

---

## 📁 Project Structure

```
forecasting_project/
├── data/
│   └── sales_data.xlsx          ← PUT YOUR EXCEL FILE HERE
├── models/
│   ├── sarima_model.py
│   ├── prophet_model.py
│   ├── xgboost_model.py
│   ├── lstm_model.py
│   ├── model_selector.py
│   └── saved/                   ← trained model files go here (auto-created)
├── utils/
│   ├── data_loader.py            ← preprocessing + feature engineering
│   └── metrics.py
├── api/
│   └── main.py                  ← FastAPI application
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_FeatureEngineering.ipynb
│   └── 03_ModelComparison.ipynb
├── outputs/                     ← metrics CSVs + plots (auto-created)
├── train_all.py                 ← master training script
├── requirements.txt
└── README.md
```

---

## ✅ Step-by-Step Setup

### STEP 1 — Place your data file

Copy your Excel file into the `data/` folder and rename it:
```
data/sales_data.xlsx
```

---

### STEP 2 — Create Python virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac / Linux
python3 -m venv venv
source venv/bin/activate
```

---

### STEP 3 — Install all dependencies

```bash
pip install -r requirements.txt
```

> ⏱ This takes 5-10 minutes the first time.  
> If Prophet fails: `pip install pystan==2.19.1.1` first, then `pip install prophet`

---

### STEP 4 — (Optional) Explore data in notebooks

```bash
pip install jupyter
jupyter notebook
```
Open and run `notebooks/01_EDA.ipynb` then `notebooks/02_FeatureEngineering.ipynb`

---

### STEP 5 — Train all models

```bash
# Full training (takes 30-90 min due to SARIMA)
python train_all.py --data data/sales_data.xlsx

# Fast training (skip SARIMA & LSTM — use for testing)
python train_all.py --data data/sales_data.xlsx --skip_sarima --skip_lstm
```

**What this does:**
- Loads & preprocesses data (resamples to weekly, handles missing values)
- Engineers lag features, rolling statistics, calendar features, holiday flags
- Trains SARIMA, Prophet, XGBoost, LSTM for ALL states
- Evaluates each model on validation set (last 8 weeks)
- Selects the best model per state (lowest RMSE)
- Saves all trained models to `models/saved/`
- Saves metrics to `outputs/`

---

### STEP 6 — Start the API

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Open your browser: **http://localhost:8000/docs**  
You'll see the interactive Swagger UI with all endpoints.

---

### STEP 7 — Test the API

**Get all states:**
```bash
curl http://localhost:8000/states
```

**Forecast next 8 weeks for a state:**
```bash
curl -X POST http://localhost:8000/forecast \
  -H "Content-Type: application/json" \
  -d '{"state": "California", "n_weeks": 8}'
```

**Force a specific model:**
```bash
curl -X POST http://localhost:8000/forecast \
  -H "Content-Type: application/json" \
  -d '{"state": "Texas", "n_weeks": 8, "model": "XGBoost"}'
```

**Bulk forecast (multiple states):**
```bash
curl -X POST http://localhost:8000/forecast/bulk \
  -H "Content-Type: application/json" \
  -d '{"states": ["California", "Texas", "Florida"], "n_weeks": 8}'
```

**View best model per state:**
```bash
curl http://localhost:8000/models/best
```

**View all metrics:**
```bash
curl http://localhost:8000/metrics
```

---

## 📡 API Endpoints Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/states` | List all states |
| POST | `/forecast` | Forecast for one state |
| POST | `/forecast/bulk` | Forecast for multiple states |
| GET | `/models/best` | Best model per state |
| GET | `/metrics` | All model evaluation metrics |
| GET | `/docs` | Swagger UI (interactive) |

---

## 🎥 Screen Recording Guide (for video submission)

Record your screen showing:
1. `train_all.py` running in terminal
2. `outputs/best_models.csv` showing which model won per state
3. API started with `uvicorn`
4. Swagger UI at `/docs` — run a live forecast
5. `notebooks/03_ModelComparison.ipynb` — show charts

Recommended tools: **Loom** (free, browser-based) or **OBS Studio**

---

## 🔧 Troubleshooting

| Problem | Fix |
|---------|-----|
| `Prophet install fails` | `pip install pystan==2.19.1.1` first |
| `TensorFlow not found` | `pip install tensorflow-cpu` (no GPU) |
| `Model not found (503)` | Run `train_all.py` first |
| `State not found (404)` | Check spelling — use `/states` to list valid names |
| `SARIMA too slow` | Use `--skip_sarima` flag during testing |

---

## 📊 Feature Engineering Summary

| Feature | Description |
|---------|-------------|
| `lag_1w` | Sales 1 week ago |
| `lag_2w` | Sales 2 weeks ago |
| `lag_4w` | Sales 4 weeks ago |
| `roll_mean_4w` | Rolling 4-week mean |
| `roll_std_4w` | Rolling 4-week std dev |
| `roll_mean_8w` | Rolling 8-week mean |
| `roll_std_8w` | Rolling 8-week std dev |
| `week_of_year` | ISO week number (1-52) |
| `month` | Month (1-12) |
| `quarter` | Quarter (1-4) |
| `day_of_week` | Day of week (0=Mon) |
| `is_holiday` | US federal holiday flag |

---

## 📈 Model Selection Logic

After training, each model is evaluated on the last 8 weeks (held-out validation set) using RMSE, MAE, and MAPE. The model with the **lowest RMSE** is automatically selected as the best model for each state. This selection is saved to `models/saved/best_models.json` and used by the API automatically.
