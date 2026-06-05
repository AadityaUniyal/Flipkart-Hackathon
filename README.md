# 🚦 Traffic Demand Prediction — Flipkart Hackathon

> Predicting urban traffic demand using AI-powered feature engineering and ensemble learning.  
> Evaluation Metric: R² Score (max score: 100)

---

## 📌 Problem Statement

Cities worldwide face growing traffic congestion that disrupts transportation and economic growth. This project builds a system that provides valuable insights into **passenger travel patterns, booking behavior, and trip cancellations** to predict **traffic demand** across urban locations and timestamps.

Given geospatial, temporal, road, and weather data, the goal is to predict the `demand` value for each entry in the test set as accurately as possible.

---

## 📂 Dataset

| File | Rows | Columns |
|---|---|---|
| `train.csv` | 77,299 | 11 (including target) |
| `test.csv` | 41,778 | 10 |
| `sample_submission.csv` | 5 | 2 |

### Columns

| Column | Description |
|---|---|
| `Index` | Unique ID |
| `geohash` | Geographic location encoded as geohash |
| `day` | Day the record was captured |
| `timestamp` | Time of record (H:MM format) |
| `RoadType` | Type of road at location |
| `NumberofLanes` | Number of lanes at location |
| `LargeVehicles` | Whether large vehicles are permitted |
| `Landmarks` | Whether landmarks are nearby |
| `Temperature` | Temperature at location |
| `Weather` | Weather condition |
| `demand` | **Target variable** — traffic demand |

---

## 🧠 Our Approach

### Feature Engineering (`feature_engineering.py`)

We engineered **40+ features** across 8 phases:

**Phase 1 — Temporal Features**
- Hour, minute, minute-of-day extracted from timestamp
- Cyclic sin/cos encodings for hour, minute, minute-of-day, and day-of-week
- Peak hour flags, rush hour flags, part-of-day (night/morning/afternoon/evening)
- Weekend flag, day-mod-7 cyclicity

**Phase 2 — Weather & Temperature**
- Temperature imputation by weather group median
- Temperature bins (cold / normal / hot)
- Temperature deviation from hourly median
- Weather anomaly detection (e.g. Sunny at night)

**Phase 3 — Geohash Features**
- Geohash decoded to latitude/longitude (custom decoder, no external library)
- KMeans spatial clustering (30 clusters)
- Character-level geohash splits (up to 6 chars)
- Frequency encoding
- OOF (Out-of-Fold) smoothed target encoding for geohash, geohash×hour, geohash×weather
- Geohash prefix frequency (3-char and 4-char)
- Location profiling: demand std, p10, p90, sum per location key

**Phase 4 — Historical Demand (Leakage-Safe)**
- `demand_prev_day`: demand at same (geohash, timestamp) from previous day
- `demand_prev_week`: demand from 7 days ago at same slot
- `demand_lag2`, `demand_lag3`: 2-day and 3-day lookbacks
- `demand_roll3`: 3-day rolling average
- All computed via day-shifted join — zero future leakage

**Phase 5 — Geo × Timestamp Mean Demand**
- `geo_ts_mean_demand`: mean demand per (geohash, timestamp) across all training days
- Computed with **leave-one-day-out** on train to prevent target leakage
- Single highest-impact feature for periodic traffic data
- `geo_hour_mean`: coarser hourly mean per geohash as robust fallback

**Phase 6 — Early Morning Features**
- Aggregated demand stats for 0:00–2:00 AM per geohash (mean, std, max, sum, last)
- Late morning slope and trend (1:00–2:00 AM velocity)
- **Spatial neighbor propagation** via BallTree (haversine) — 8 nearest neighbors
- Temporal decay from 2:00 AM onward: `early_decay = exp(-elapsed / 120)`
- Decayed versions of early morning stats
- Day detection is dynamic (last 2 training days) — not hardcoded

**Phase 7 — Interaction Features**
- Road × lanes, weather × hour, geo × hour
- Landmark × road, weather × temperature bin
- Vehicle × road, vehicle × lanes, road × lanes × vehicles

**Phase 8 — Encoding & Memory Reduction**
- Label encoding for all categorical columns (fit on train+test combined vocab)
- Downcast float64 → float32, int64 → int32 (~50% memory reduction)

---

### Models (`train_model.py`)

We train **5 diverse base models**:

| Model | Objective | Notes |
|---|---|---|
| CatBoost | RMSE | Early stopping, depth 4–10 |
| LightGBM | Tweedie | Early stopping, num_leaves 20–200 |
| XGBoost | Tweedie | Early stopping, max_depth 4–12 |
| ExtraTrees | MSE | 300 trees, max_depth 12 |
| KNeighbors | Distance-weighted | StandardScaler pipeline, k=15 |

**Hyperparameter Tuning**: Optuna with 100 trials per boosting model, maximising validation R².

---

### Ensemble & Stacking (`train_model.py`, `predict.py`)

- **TimeSeriesSplit (5 folds)** for OOF stacking — prevents future data leakage
- Ridge meta-model trained on OOF predictions of all 5 base models
- **Nelder-Mead weight optimisation** on validation set for weighted blending
- Final prediction = Ridge stacking output, clipped to training demand range

---

### Validation Strategy (`run_pipeline.py`)

- Time-based train/validation split (last 20% of timestamps as validation)
- Models trained on earlier data, evaluated on later data
- Best iteration counts from val-split used for full-data retrain
- Full pipeline: load → engineer → tune → train → stack → submit

---

## 📈 Expected Performance

| Stage | Estimated R² | Score |
|---|---|---|
| Baseline (no features) | ~0.60 | 60 |
| After geo_ts_mean_demand | ~0.92 | 92 |
| After lag features | ~0.96 | 96 |
| After full pipeline + tuning | ~0.98–0.99 | 98–99 |

---

## 🗂️ Project Structure

```
Flipkart-Hackathon/
├── data/                        ← place train.csv and test.csv here (not committed)
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── feature_engineering.py       ← all feature engineering (8 phases, 40+ features)
├── train_model.py               ← model training, Optuna tuning, stacking
├── predict.py                   ← inference, submission creation, validation
├── run_pipeline.py              ← master orchestration script
├── requirements.txt             ← Python dependencies
├── .gitignore
└── README.md
```

---

## 🚀 How to Run

**1. Clone the repo**
```bash
git clone https://github.com/AadityaUniyal/Flipkart-Hackathon.git
cd Flipkart-Hackathon
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Add data files**
Place `train.csv`, `test.csv`, and `sample_submission.csv` inside a `data/` folder.

**4. Run the full pipeline**
```bash
python run_pipeline.py
```

To skip Optuna tuning (faster run):
```bash
python run_pipeline.py --no-tune
```

**5. Submit**
Upload the generated `submission.csv` to the hackathon platform.

---

## ✅ Submission Format

| Column | Type | Description |
|---|---|---|
| `Index` | int | Matches test.csv Index exactly |
| `demand` | float | Predicted traffic demand |

- Shape: exactly **41,778 × 2**
- No null values
- File format: `.csv`

---

## 🛠️ Requirements

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
catboost>=1.0.0
lightgbm>=3.3.0
xgboost>=1.6.0
optuna>=3.0.0
scipy>=1.7.0
```

---

## 👤 Author

**Aaditya Uniyal**  
Flipkart Hackathon — Traffic Demand Prediction
