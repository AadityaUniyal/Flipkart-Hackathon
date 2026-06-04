"""
Traffic Demand Prediction – Model Training & Tuning
====================================================
Phase 14: Tweedie regression, ensembling expansion (ExtraTrees & KNN), K-Fold Stacking
"""

import numpy as np
from sklearn.metrics import r2_score
from catboost import CatBoostRegressor
import lightgbm as lgb
import xgboost as xgb
import optuna
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ═══════════════════════════════════════════════════════════════════
# Baseline Training
# ═══════════════════════════════════════════════════════════════════

def train_catboost(X_tr, y_tr, X_val, y_val, extra_params=None):
    """Train a CatBoost regressor with early stopping."""
    params = dict(
        iterations=3000,
        depth=8,
        learning_rate=0.05,
        l2_leaf_reg=3,
        random_seed=42,
        verbose=200,
        early_stopping_rounds=300,
        loss_function="RMSE",
    )
    if extra_params:
        params.update(extra_params)
    model = CatBoostRegressor(**params)
    model.fit(X_tr, y_tr, eval_set=(X_val, y_val))
    return model


def train_lightgbm(X_tr, y_tr, X_val, y_val, extra_params=None):
    """Train a LightGBM regressor with early stopping using Tweedie objective."""
    params = dict(
        objective="tweedie",
        metric="rmse",
        num_leaves=63,
        max_depth=8,
        learning_rate=0.05,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        n_estimators=3000,
        random_state=42,
        verbose=-1,
    )
    if extra_params:
        params.update(extra_params)
    callbacks = [lgb.early_stopping(300, verbose=False), lgb.log_evaluation(200)]
    model = lgb.LGBMRegressor(**params)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=callbacks)
    return model


def train_xgboost(X_tr, y_tr, X_val, y_val, extra_params=None):
    """Train an XGBoost regressor with early stopping using Tweedie objective."""
    params = dict(
        objective="reg:tweedie",
        eval_metric="rmse",
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        n_estimators=3000,
        random_state=42,
        verbosity=0,
        early_stopping_rounds=300,
    )
    if extra_params:
        params.update(extra_params)
    model = xgb.XGBRegressor(**params)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=200)
    return model


def train_extratrees(X_tr, y_tr, X_val, y_val, extra_params=None):
    """Train an ExtraTrees regressor."""
    params = dict(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1,
    )
    if extra_params:
        params.update(extra_params)
    model = ExtraTreesRegressor(**params)
    model.fit(X_tr, y_tr)
    return model


def train_knn(X_tr, y_tr, X_val, y_val, extra_params=None):
    """Train a KNeighbors regressor with scaling pipeline."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    params = dict(
        n_neighbors=15,
        weights="distance",
        n_jobs=-1,
    )
    if extra_params:
        params.update(extra_params)
    model = make_pipeline(StandardScaler(), KNeighborsRegressor(**params))
    model.fit(X_tr, y_tr)
    return model


# ═══════════════════════════════════════════════════════════════════
# Optuna Hyperparameter Tuning
# ═══════════════════════════════════════════════════════════════════

def _obj_catboost(trial, X_tr, y_tr, X_val, y_val):
    p = dict(
        iterations=2000,
        verbose=0,
        random_seed=42,
        early_stopping_rounds=150,
        loss_function="RMSE",
        depth=trial.suggest_int("depth", 4, 10),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        min_data_in_leaf=trial.suggest_int("min_data_in_leaf", 5, 100),
    )
    m = CatBoostRegressor(**p)
    m.fit(X_tr, y_tr, eval_set=(X_val, y_val))
    return r2_score(y_val, m.predict(X_val))


def _obj_lgbm(trial, X_tr, y_tr, X_val, y_val):
    p = dict(
        objective="tweedie",
        metric="rmse",
        n_estimators=2000,
        verbose=-1,
        random_state=42,
        num_leaves=trial.suggest_int("num_leaves", 20, 200),
        max_depth=trial.suggest_int("max_depth", 4, 12),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        min_child_samples=trial.suggest_int("min_child_samples", 5, 100),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 0.0, 10.0),
        reg_lambda=trial.suggest_float("reg_lambda", 0.0, 10.0),
    )
    cb = [lgb.early_stopping(150, verbose=False)]
    m = lgb.LGBMRegressor(**p)
    m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=cb)
    return r2_score(y_val, m.predict(X_val))


def _obj_xgb(trial, X_tr, y_tr, X_val, y_val):
    p = dict(
        objective="reg:tweedie",
        eval_metric="rmse",
        n_estimators=2000,
        verbosity=0,
        random_state=42,
        early_stopping_rounds=150,
        max_depth=trial.suggest_int("max_depth", 4, 12),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 0.0, 10.0),
        reg_lambda=trial.suggest_float("reg_lambda", 0.0, 10.0),
        min_child_weight=trial.suggest_int("min_child_weight", 1, 30),
    )
    m = xgb.XGBRegressor(**p)
    m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    return r2_score(y_val, m.predict(X_val))


def tune_models(X_tr, y_tr, X_val, y_val, n_trials=30):
    """Run Optuna for gradient boosted trees; return dict of best params."""
    best = {}
    for name, fn in [
        ("catboost", _obj_catboost),
        ("lightgbm", _obj_lgbm),
        ("xgboost", _obj_xgb),
    ]:
        print(f"  Tuning {name} ({n_trials} trials) ...")
        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial, _fn=fn: _fn(trial, X_tr, y_tr, X_val, y_val),
            n_trials=n_trials,
        )
        best[name] = study.best_params
        print(f"    Best R2 = {study.best_value:.6f}")
        print(f"    Params  = {study.best_params}")
    
    # ExtraTrees and KNN do not require extensive Optuna tuning
    best["extratrees"] = {}
    best["knn"] = {}
    return best


# ═══════════════════════════════════════════════════════════════════
# Ensemble Weight Optimisation & Stacking
# ═══════════════════════════════════════════════════════════════════

def optimize_weights(preds_list, y_true, init=None):
    """Find weights that maximise validation R²."""
    from scipy.optimize import minimize as sp_min

    n = len(preds_list)
    x0 = init if init is not None else [1.0 / n] * n

    def neg_r2(w):
        w = np.abs(w)
        w = w / w.sum()
        blend = sum(w[i] * preds_list[i] for i in range(n))
        return -r2_score(y_true, blend)

    res = sp_min(neg_r2, x0, method="Nelder-Mead")
    w = np.abs(res.x)
    return w / w.sum()


def train_stacking_meta_model(X, y, cb_params, lgb_params, xgb_params, et_params, knn_params, cb_iters, lgb_iters, xgb_iters):
    """Perform 5-Fold CV to generate OOF predictions for 5 base models and train a Ridge meta-model."""
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros((len(X), 5))
    
    cb_p = dict(
        iterations=cb_iters,
        depth=(cb_params or {}).get("depth", 8),
        learning_rate=(cb_params or {}).get("learning_rate", 0.05),
        l2_leaf_reg=(cb_params or {}).get("l2_leaf_reg", 3),
        random_seed=42,
        verbose=0,
        loss_function="RMSE",
    )
    if cb_params and "min_data_in_leaf" in cb_params:
        cb_p["min_data_in_leaf"] = cb_params["min_data_in_leaf"]

    lgb_p = dict(
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

    xgb_p = dict(
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

    et_p = dict(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1,
    )
    if et_params:
        et_p.update(et_params)

    knn_p = dict(
        n_neighbors=15,
        weights="distance",
        n_jobs=-1,
    )
    if knn_params:
        knn_p.update(knn_params)
    
    print("\n  Generating OOF predictions for Stacking (5 models)...")
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_val = X.iloc[val_idx]
        
        # 1. CatBoost
        cb = CatBoostRegressor(**cb_p)
        cb.fit(X_tr, y_tr)
        oof_preds[val_idx, 0] = cb.predict(X_val)
        
        # 2. LightGBM
        lgb_m = lgb.LGBMRegressor(**lgb_p)
        lgb_m.fit(X_tr, y_tr)
        oof_preds[val_idx, 1] = lgb_m.predict(X_val)
        
        # 3. XGBoost
        xgb_m = xgb.XGBRegressor(**xgb_p)
        xgb_m.fit(X_tr, y_tr)
        oof_preds[val_idx, 2] = xgb_m.predict(X_val)
        
        # 4. ExtraTrees
        et_m = ExtraTreesRegressor(**et_p)
        et_m.fit(X_tr, y_tr)
        oof_preds[val_idx, 3] = et_m.predict(X_val)
        
        # 5. KNeighbors
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        knn_m = make_pipeline(StandardScaler(), KNeighborsRegressor(**knn_p))
        knn_m.fit(X_tr, y_tr)
        oof_preds[val_idx, 4] = knn_m.predict(X_val)
        
        print(f"    Fold {fold+1} complete")
        
    # Fit Ridge meta-model
    meta_model = Ridge(alpha=1.0, fit_intercept=True)
    meta_model.fit(oof_preds, y)
    
    print(f"  Stacking Ridge coefficients: CB={meta_model.coef_[0]:.3f}, LGB={meta_model.coef_[1]:.3f}, XGB={meta_model.coef_[2]:.3f}, ET={meta_model.coef_[3]:.3f}, KNN={meta_model.coef_[4]:.3f}")
    print(f"  Intercept: {meta_model.intercept_:.6f}")
    
    # Print R2 score of stacking OOF
    oof_r2 = r2_score(y, meta_model.predict(oof_preds))
    print(f"  Stacking OOF R2 = {oof_r2:.6f}")
    
    return meta_model
