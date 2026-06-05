"""
Traffic Demand Prediction – Prediction & Submission
====================================================
Phase 14: Generate stacked predictions with 5 models
"""

import numpy as np
import pandas as pd

DEMAND_MIN = 6.245650e-7
DEMAND_MAX = 1.0


def predict_ensemble(models, weights, X):
    """Weighted-average prediction from a list of models."""
    preds = [m.predict(X) for m in models]
    blend = sum(weights[i] * preds[i] for i in range(len(models)))
    # Clip to valid demand range (0, 1]
    return np.clip(blend, DEMAND_MIN, DEMAND_MAX)


def predict_stacking(models, meta_model, X):
    """Stacking prediction from 5 base models and Ridge meta-model."""
    preds = np.zeros((len(X), 5))
    preds[:, 0] = models[0].predict(X)
    preds[:, 1] = models[1].predict(X)
    preds[:, 2] = models[2].predict(X)
    preds[:, 3] = models[3].predict(X)
    preds[:, 4] = models[4].predict(X)
    blend = meta_model.predict(preds)
    # Clip to valid demand range (0, 1]
    return np.clip(blend, DEMAND_MIN, DEMAND_MAX)


def create_submission(test_df, predictions, output_path, train_raw=None):
    """Save ``submission.csv`` in the competition format."""
    demand_min = train_raw["demand"].min() if train_raw is not None else 1e-7
    demand_max = train_raw["demand"].max() if train_raw is not None else 1.0
    predictions = np.clip(predictions, demand_min, demand_max)

    sub = pd.DataFrame({
        "Index": test_df["Index"],
        "demand": predictions,
    })

    if train_raw is not None:
        geo_ts_max = (
            train_raw.groupby(["geohash", "timestamp"])["demand"]
            .max()
            .rename("max_train_demand")
            .reset_index()
        )
        bounds = test_df[["Index", "geohash", "timestamp"]].merge(
            geo_ts_max, on=["geohash", "timestamp"], how="left"
        )
        upper = (bounds["max_train_demand"] * 1.2).fillna(demand_max)
        sub["demand"] = np.minimum(sub["demand"].to_numpy(), upper.to_numpy())
        sub["demand"] = np.clip(sub["demand"], demand_min, demand_max)

    assert sub.shape == (41778, 2), f"WRONG SHAPE: {sub.shape}"
    assert list(sub.columns) == ["Index", "demand"], (
        f"WRONG COLUMNS: {sub.columns.tolist()}"
    )
    assert sub["Index"].tolist() == test_df["Index"].tolist(), (
        "Index values do not match test data"
    )
    assert sub["demand"].isna().sum() == 0, "NaN values in demand column"
    sub.to_csv(output_path, index=False)
    print(f"\n  Submission saved  -> {output_path}")
    print(f"  Shape             : {sub.shape}")
    print(f"  Demand range      : [{sub['demand'].min():.6f} , {sub['demand'].max():.6f}]")
    print(f"  Demand mean       : {sub['demand'].mean():.6f}")
    return sub
