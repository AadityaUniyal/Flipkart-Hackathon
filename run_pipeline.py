"""
Traffic Demand Prediction – Master Pipeline
=============================================
Single entry-point that orchestrates the full workflow:

  1. Load data
  2. Feature engineering
  3. Validation split & baseline evaluation
  4. Hyperparameter tuning (Optuna)
  5. Retrain on validation split
  6. Ensemble weight optimization
  7. Retrain on full data
  8. Stacking prediction → submission.csv
"""

import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

# ── Local modules ──────────────────────────────────────────────────
from feature_engineering import engineer_all_features, get_feature_cols
from train_model import (
    train_catboost,
    train_lightgbm,
    train_xgboost,
    train_extratrees,
    train_knn,
    tune_models,
    optimize_weights,
    train_stacking_meta_model,
)
from predict import predict_ensemble, predict_stacking, create_submission

# ── Configuration ──────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
TRAIN_CSV  = os.path.join(DATA_DIR, "train.csv")
TEST_CSV   = os.path.join(DATA_DIR, "test.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "submission.csv")

OPTUNA_TRIALS = 30          # Trials per model during tuning
VAL_RATIO     = 0.20        # Last 20 % of timestamps → validation


# ═══════════════════════════════════════════════════════════════════

def main(do_tune: bool = True):
    t0 = time.time()

    # ────────────────────────────────────────────────────────────────
    # 1. Load data
    # ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Step 1 : Loading Data")
    print("=" * 60)
    train_raw = pd.read_csv(TRAIN_CSV)
    test_raw  = pd.read_csv(TEST_CSV)
    print(f"  Train  : {train_raw.shape}")
    print(f"  Test   : {test_raw.shape}")

    # ────────────────────────────────────────────────────────────────
    # 2. Feature engineering
    # ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Step 2 : Feature Engineering")
    print("=" * 60)
    train_df, test_df = engineer_all_features(train_raw.copy(), test_raw.copy())
    # Filter out early morning rows (<= 2:00 AM) to prevent target leakage
    train_df = train_df[train_df["minute_of_day"] > 120].reset_index(drop=True)
    feature_cols = get_feature_cols(train_df)
    print(f"  Feature count : {len(feature_cols)}")
    print(f"  Features      : {feature_cols[:10]} ...")

    # ────────────────────────────────────────────────────────────────
    # 3. Validation split  (time-based within Day 48)
    # ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Step 3 : Validation Split")
    print("=" * 60)
    train_df_sorted = train_df.sort_values("minute_of_day").reset_index(drop=True)
    split_idx = int(len(train_df_sorted) * (1 - VAL_RATIO))
    df_tr  = train_df_sorted.iloc[:split_idx]
    df_val = train_df_sorted.iloc[split_idx:]

    X_tr,  y_tr  = df_tr[feature_cols],  df_tr["demand"]
    X_val, y_val = df_val[feature_cols], df_val["demand"]
    print(f"  Train split : {X_tr.shape}")
    print(f"  Val   split : {X_val.shape}")

    # ────────────────────────────────────────────────────────────────
    # 4. Hyperparameter tuning  (optional)
    # ────────────────────────────────────────────────────────────────
    cb_params = lgb_params = xgb_params = None
    if do_tune:
        print("\n" + "=" * 60)
        print(f"  Step 4 : Optuna Tuning  ({OPTUNA_TRIALS} trials / model)")
        print("=" * 60)
        best = tune_models(X_tr, y_tr, X_val, y_val, n_trials=OPTUNA_TRIALS)
        cb_params  = best["catboost"]
        lgb_params = best["lightgbm"]
        xgb_params = best["xgboost"]
    else:
        print("\n  Step 4 : Skipped (--no-tune)")

    # ────────────────────────────────────────────────────────────────
    # 5. Train on validation split  (to find best #iterations)
    # ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Step 5 : Training on Validation Split")
    print("=" * 60)

    print("\n  -> CatBoost")
    cb_model  = train_catboost(X_tr, y_tr, X_val, y_val, cb_params)
    print("\n  -> LightGBM")
    lgb_model = train_lightgbm(X_tr, y_tr, X_val, y_val, lgb_params)
    print("\n  -> XGBoost")
    xgb_model = train_xgboost(X_tr, y_tr, X_val, y_val, xgb_params)
    print("\n  -> ExtraTrees")
    et_model  = train_extratrees(X_tr, y_tr, X_val, y_val)
    print("\n  -> KNeighbors")
    knn_model = train_knn(X_tr, y_tr, X_val, y_val)

    models = [cb_model, lgb_model, xgb_model, et_model, knn_model]
    names  = ["CatBoost", "LightGBM", "XGBoost", "ExtraTrees", "KNeighbors"]

    # Per-model validation R²
    val_preds = []
    print()
    for name, m in zip(names, models):
        p = m.predict(X_val)
        val_preds.append(p)
        print(f"  {name:10s}  val R2 = {r2_score(y_val, p):.6f}")

    # ────────────────────────────────────────────────────────────────
    # 6. Ensemble weight optimisation
    # ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Step 6 : Ensemble Weight Optimisation")
    print("=" * 60)
    weights = optimize_weights(val_preds, y_val.values, [0.3, 0.3, 0.2, 0.1, 0.1])
    blend_val = sum(weights[i] * val_preds[i] for i in range(5))
    print(f"  Weights   : CB={weights[0]:.3f}  LGB={weights[1]:.3f}  XGB={weights[2]:.3f}  ET={weights[3]:.3f}  KNN={weights[4]:.3f}")
    print(f"  Ensemble  val R2 = {r2_score(y_val, blend_val):.6f}")

    # ────────────────────────────────────────────────────────────────
    # 7. Retrain on FULL training data
    # ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Step 7 : Retraining on Full Training Data")
    print("=" * 60)

    X_full = train_df[feature_cols]
    y_full = train_df["demand"]

    # Retrieve best iteration counts from the val-split run
    cb_iters  = cb_model.tree_count_
    lgb_iters = getattr(lgb_model, "best_iteration_", lgb_model.n_estimators)
    xgb_iters = getattr(xgb_model, "best_iteration", xgb_model.n_estimators) or xgb_model.n_estimators

    print(f"  Best iters: CB={cb_iters}  LGB={lgb_iters}  XGB={xgb_iters}")

    # CatBoost – full retrain ─────────────────────────────────────
    from catboost import CatBoostRegressor as CBR

    final_cb_p = dict(
        iterations=cb_iters,
        depth=(cb_params or {}).get("depth", 8),
        learning_rate=(cb_params or {}).get("learning_rate", 0.05),
        l2_leaf_reg=(cb_params or {}).get("l2_leaf_reg", 3),
        random_seed=42,
        verbose=200,
        loss_function="RMSE",
    )
    if cb_params and "min_data_in_leaf" in cb_params:
        final_cb_p["min_data_in_leaf"] = cb_params["min_data_in_leaf"]
    print("\n  -> CatBoost (full)")
    cb_final = CBR(**final_cb_p)
    cb_final.fit(X_full, y_full)

    # LightGBM – full retrain ─────────────────────────────────────
    import lightgbm as _lgb

    final_lgb_p = dict(
        objective="tweedie",
        metric="rmse",
        n_estimators=lgb_iters,
        num_leaves=(lgb_params or {}).get("num_leaves", 63),
        max_depth=(lgb_params or {}).get("max_depth", 8),
        learning_rate=(lgb_params or {}).get("learning_rate", 0.05),
        min_child_samples=(lgb_params or {}).get("min_child_samples", 20),
        subsample=(lgb_params or {}).get("subsample", 0.8),
        colsample_bytree=(lgb_params or {}).get("colsample_bytree", 0.8),
        reg_alpha=(lgb_params or {}).get("reg_alpha", 0.0),
        reg_lambda=(lgb_params or {}).get("reg_lambda", 0.0),
        random_state=42,
        verbose=-1,
    )
    print("  -> LightGBM (full)")
    lgb_final = _lgb.LGBMRegressor(**final_lgb_p)
    lgb_final.fit(X_full, y_full)

    # XGBoost – full retrain ──────────────────────────────────────
    import xgboost as _xgb

    final_xgb_p = dict(
        objective="reg:tweedie",
        eval_metric="rmse",
        n_estimators=xgb_iters,
        max_depth=(xgb_params or {}).get("max_depth", 8),
        learning_rate=(xgb_params or {}).get("learning_rate", 0.05),
        subsample=(xgb_params or {}).get("subsample", 0.8),
        colsample_bytree=(xgb_params or {}).get("colsample_bytree", 0.8),
        reg_alpha=(xgb_params or {}).get("reg_alpha", 0.0),
        reg_lambda=(xgb_params or {}).get("reg_lambda", 0.0),
        min_child_weight=(xgb_params or {}).get("min_child_weight", 1),
        random_state=42,
        verbosity=0,
    )
    print("  -> XGBoost (full)")
    xgb_final = _xgb.XGBRegressor(**final_xgb_p)
    xgb_final.fit(X_full, y_full)

    # ExtraTrees – full retrain ───────────────────────────────────
    from sklearn.ensemble import ExtraTreesRegressor
    print("  -> ExtraTrees (full)")
    et_final = ExtraTreesRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1,
    )
    et_final.fit(X_full, y_full)

    # KNeighbors – full retrain ───────────────────────────────────
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    print("  -> KNeighbors (full)")
    knn_final = make_pipeline(
        StandardScaler(),
        KNeighborsRegressor(
            n_neighbors=15,
            weights="distance",
            n_jobs=-1,
        )
    )
    knn_final.fit(X_full, y_full)

    final_models = [cb_final, lgb_final, xgb_final, et_final, knn_final]

    # Stacking – meta model training ──────────────────────────────
    meta_model = train_stacking_meta_model(
        X_full, y_full,
        cb_params, lgb_params, xgb_params, {}, {},
        cb_iters, lgb_iters, xgb_iters
    )

    # ────────────────────────────────────────────────────────────────
    # 8. Predict & create submission
    # ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Step 8 : Generating Submission")
    print("=" * 60)
    X_test = test_df[feature_cols]
    preds  = predict_stacking(final_models, meta_model, X_test)
    sub    = create_submission(test_df, preds, OUTPUT_CSV)

    # ── Summary ────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print(f"  * Pipeline complete in {elapsed / 60:.1f} min")
    print(f"  * Validation R2 (ensemble) = {r2_score(y_val, blend_val):.6f}")
    print(f"  * Submission: {OUTPUT_CSV}  ({len(sub)} rows)")
    print("=" * 60 + "\n")


# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    tune = "--no-tune" not in sys.argv
    main(do_tune=tune)
