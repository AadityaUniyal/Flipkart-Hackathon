import numpy as np
import pandas as pd


def predict_ensemble(models, weights, X):
    """Weighted-average prediction from a list of models."""
    preds = [m.predict(X) for m in models]
    blend = sum(weights[i] * preds[i] for i in range(len(models)))
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
    return np.clip(blend, 1e-7, 1.0)


def create_submission(test_df, predictions, output_path, train_raw=None):
    """Save submission.csv in the competition format."""
    # Use actual demand range from training data if available
    if train_raw is not None:
        d_min = float(train_raw["demand"].min())
        d_max = float(train_raw["demand"].max())
        predictions = np.clip(predictions, d_min, d_max)

    sub = pd.DataFrame({
        "Index": test_df["Index"],
        "demand": predictions,
    })

    # Hard validation — will crash loudly if something is wrong
    assert sub.shape == (41778, 2), f"WRONG SHAPE: {sub.shape}"
    assert list(sub.columns) == ["Index", "demand"], f"WRONG COLUMNS: {sub.columns.tolist()}"
    assert sub["demand"].isna().sum() == 0, "NaN values in demand column!"

    sub.to_csv(output_path, index=False)
    print(f"\n  Submission saved -> {output_path}")
    print(f"  Shape  : {sub.shape}")
    print(f"  Demand : [{sub['demand'].min():.6f}, {sub['demand'].max():.6f}]")
    print(f"  Mean   : {sub['demand'].mean():.6f}")
    return sub
