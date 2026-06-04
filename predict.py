"""
Traffic Demand Prediction – Prediction & Submission
====================================================
Phase 14: Generate stacked predictions with 5 models
"""

import numpy as np
import pandas as pd


def predict_ensemble(models, weights, X):
    """Weighted-average prediction from a list of models."""
    preds = [m.predict(X) for m in models]
    blend = sum(weights[i] * preds[i] for i in range(len(models)))
    # Clip to valid demand range (0, 1]
    return np.clip(blend, 1e-7, 1.0)


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
    return np.clip(blend, 1e-7, 1.0)


def create_submission(test_df, predictions, output_path):
    """Save ``submission.csv`` in the competition format."""
    sub = pd.DataFrame({
        "Index": test_df["Index"],
        "demand": predictions,
    })
    sub.to_csv(output_path, index=False)
    print(f"\n  Submission saved  -> {output_path}")
    print(f"  Shape             : {sub.shape}")
    print(f"  Demand range      : [{sub['demand'].min():.6f} , {sub['demand'].max():.6f}]")
    print(f"  Demand mean       : {sub['demand'].mean():.6f}")
    return sub
