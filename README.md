# Flipkart Traffic Demand Prediction

An ensemble machine-learning pipeline for predicting traffic demand from temporal,
spatial, weather, and road-network features. The project combines CatBoost,
LightGBM, XGBoost, Extra Trees, and K-Nearest Neighbors, then blends their
predictions with a Ridge stacking model.

## Project Structure

```text
.
|-- data/                         # Training, test, and sample submission CSVs
|-- scratch/                      # Experimental validation and modeling scripts
|-- feature_engineering.py        # Feature creation and preprocessing
|-- train_model.py                # Model training, tuning, and stacking
|-- predict.py                    # Ensemble prediction and submission helpers
|-- run_pipeline.py               # End-to-end pipeline entry point
`-- Traffic_Demand_Hackathon_Plan.txt
```

## Setup

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
```

Activate the environment:

```bash
# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Usage

Run the complete pipeline with Optuna hyperparameter tuning:

```bash
python run_pipeline.py
```

Skip tuning for a faster run using the default model parameters:

```bash
python run_pipeline.py --no-tune
```

The pipeline reads `data/train.csv` and `data/test.csv`, then writes predictions
to `submission.csv`.

## Approach

- Temporal, cyclic, geospatial, weather, and interaction feature engineering
- Time-aware validation to reflect the competition's distribution shift
- Optuna hyperparameter tuning
- CatBoost, LightGBM, XGBoost, Extra Trees, and KNN base models
- Weighted ensembling and Ridge stacking
- Prediction clipping to the valid demand range

## License

This project is licensed under the [MIT License](LICENSE).
