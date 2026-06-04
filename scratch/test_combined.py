import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

def clean_raw(df):
    df["RoadType"]      = df["RoadType"].fillna("Unknown")
    df["Weather"]       = df["Weather"].fillna("Unknown")
    df["LargeVehicles"] = df["LargeVehicles"].fillna("Unknown")
    df["Landmarks"]     = df["Landmarks"].fillna("Unknown")
    df["NumberofLanes"]  = df["NumberofLanes"].fillna(df["NumberofLanes"].median())
    return df

def add_temporal_features(df):
    parts = df["timestamp"].str.split(":", expand=True).astype(int)
    df["hour"]          = parts[0]
    df["minute"]        = parts[1]
    df["minute_of_day"] = df["hour"] * 60 + df["minute"]

    df["part_of_day"] = np.select(
        [df["hour"].between(0, 5),
         df["hour"].between(6, 11),
         df["hour"].between(12, 17),
         df["hour"].between(18, 23)],
        [0, 1, 2, 3], default=0,
    )

    df["is_peak_hour"] = df["hour"].isin([7, 8, 9, 17, 18, 19]).astype(int)
    df["is_rush_hour"] = df["hour"].isin(
        list(range(6, 11)) + list(range(16, 21))
    ).astype(int)

    df["sin_hour"]  = np.sin(2 * np.pi * df["hour"] / 24)
    df["cos_hour"]  = np.cos(2 * np.pi * df["hour"] / 24)
    df["sin_min"]   = np.sin(2 * np.pi * df["minute"] / 60)
    df["cos_min"]   = np.cos(2 * np.pi * df["minute"] / 60)
    df["sin_mod"]   = np.sin(2 * np.pi * df["minute_of_day"] / 1440)
    df["cos_mod"]   = np.cos(2 * np.pi * df["minute_of_day"] / 1440)
    return df

def add_weather_features(df):
    df["is_day"] = ((df["hour"] >= 6) & (df["hour"] < 18)).astype(int)
    df["weather_anomaly"] = ((df["Weather"] == "Sunny") & (df["is_day"] == 0)).astype(int)
    weather_med = df.groupby("Weather")["Temperature"].transform("median")
    df["Temperature"] = df["Temperature"].fillna(weather_med)
    df["Temperature"] = df["Temperature"].fillna(df["Temperature"].median())
    df["temp_bin"] = np.select(
        [df["Temperature"] < 15,
         df["Temperature"].between(15, 30),
         df["Temperature"] > 30],
        [0, 1, 2], default=1,
    )
    return df

def add_spatial_temporal_features(train_df, test_df):
    global_mean = train_df["demand"].mean()

    # Define loc_key and char splits
    for df in (train_df, test_df):
        df["loc_key"] = (
            df["geohash"].astype(str) + "_" +
            df["RoadType"].astype(str) + "_" +
            df["NumberofLanes"].astype(str) + "_" +
            df["LargeVehicles"].astype(str) + "_" +
            df["Landmarks"].astype(str)
        )
        max_len = min(6, df["geohash"].str.len().min())
        for i in range(max_len):
            df[f"geo_char_{i+1}"] = df["geohash"].str[i]

    # Frequency encoding
    geo_freq = train_df["geohash"].value_counts().to_dict()
    loc_freq = train_df["loc_key"].value_counts().to_dict()
    
    for df in (train_df, test_df):
        df["geohash_freq"] = df["geohash"].map(geo_freq).fillna(0)
        df["loc_freq"]     = df["loc_key"].map(loc_freq).fillna(0)

    # 5-fold cross-validation for OOF target encodings
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Initialize OOF columns
    target_cols = [
        "geohash_te", "geo_hour_demand", "geo_weather_demand",
        "loc_te", "loc_hour_te", "loc_weather_te"
    ]
    for col in target_cols:
        train_df[col] = np.nan

    m1, m2, m3, m4 = 20, 10, 5, 5

    for tr_idx, val_idx in kf.split(train_df):
        tr_fold = train_df.iloc[tr_idx]
        
        # 1. geohash_te
        geo_stats = tr_fold.groupby("geohash")["demand"].agg(["mean", "count"])
        geo_te = ((geo_stats["count"] * geo_stats["mean"] + m1 * global_mean) / (geo_stats["count"] + m1)).to_dict()
        train_df.iloc[val_idx, train_df.columns.get_loc("geohash_te")] = train_df.iloc[val_idx]["geohash"].map(geo_te)
        
        # 2. geo_hour_demand
        gh_stats = tr_fold.groupby(["geohash", "hour"])["demand"].mean().reset_index(name="val")
        val_slice = train_df.iloc[val_idx][["geohash", "hour"]]
        val_merged = val_slice.merge(gh_stats, on=["geohash", "hour"], how="left")
        val_merged["val"] = val_merged["val"].fillna(pd.Series(train_df.iloc[val_idx]["geohash"].map(geo_te).fillna(global_mean).values, index=val_merged.index))
        train_df.iloc[val_idx, train_df.columns.get_loc("geo_hour_demand")] = val_merged["val"].values

        # 3. geo_weather_demand
        gw_stats = tr_fold.groupby(["geohash", "Weather"])["demand"].mean().reset_index(name="val")
        val_slice = train_df.iloc[val_idx][["geohash", "Weather"]]
        val_merged = val_slice.merge(gw_stats, on=["geohash", "Weather"], how="left")
        val_merged["val"] = val_merged["val"].fillna(pd.Series(train_df.iloc[val_idx]["geohash"].map(geo_te).fillna(global_mean).values, index=val_merged.index))
        train_df.iloc[val_idx, train_df.columns.get_loc("geo_weather_demand")] = val_merged["val"].values

        # 4. loc_te (smooth towards geohash_te)
        loc_stats = tr_fold.groupby("loc_key")["demand"].agg(["mean", "count"])
        loc_to_geo = tr_fold.groupby("loc_key")["geohash"].first().to_dict()
        loc_stats["geo_te"] = loc_stats.index.map(loc_to_geo).map(geo_te).fillna(global_mean)
        loc_te = ((loc_stats["count"] * loc_stats["mean"] + m2 * loc_stats["geo_te"]) / (loc_stats["count"] + m2)).to_dict()
        train_df.iloc[val_idx, train_df.columns.get_loc("loc_te")] = train_df.iloc[val_idx]["loc_key"].map(loc_te)

        # 5. loc_hour_te (smooth towards loc_te)
        lh_stats = tr_fold.groupby(["loc_key", "hour"])["demand"].agg(["mean", "count"]).reset_index()
        lh_stats["loc_te"] = lh_stats["loc_key"].map(loc_te).fillna(global_mean)
        lh_stats["te"] = (lh_stats["count"] * lh_stats["mean"] + m3 * lh_stats["loc_te"]) / (lh_stats["count"] + m3)
        
        val_slice = train_df.iloc[val_idx][["loc_key", "hour"]]
        val_merged = val_slice.merge(lh_stats[["loc_key", "hour", "te"]], on=["loc_key", "hour"], how="left")
        val_merged["te"] = val_merged["te"].fillna(pd.Series(train_df.iloc[val_idx]["loc_key"].map(loc_te).fillna(global_mean).values, index=val_merged.index))
        train_df.iloc[val_idx, train_df.columns.get_loc("loc_hour_te")] = val_merged["te"].values

        # 6. loc_weather_te (smooth towards loc_te)
        lw_stats = tr_fold.groupby(["loc_key", "Weather"])["demand"].agg(["mean", "count"]).reset_index()
        lw_stats["loc_te"] = lw_stats["loc_key"].map(loc_te).fillna(global_mean)
        lw_stats["te"] = (lw_stats["count"] * lw_stats["mean"] + m4 * lw_stats["loc_te"]) / (lw_stats["count"] + m4)
        
        val_slice = train_df.iloc[val_idx][["loc_key", "Weather"]]
        val_merged = val_slice.merge(lw_stats[["loc_key", "Weather", "te"]], on=["loc_key", "Weather"], how="left")
        val_merged["te"] = val_merged["te"].fillna(pd.Series(train_df.iloc[val_idx]["loc_key"].map(loc_te).fillna(global_mean).values, index=val_merged.index))
        train_df.iloc[val_idx, train_df.columns.get_loc("loc_weather_te")] = val_merged["te"].values

    # Fill residual NaNs
    train_df["geohash_te"] = train_df["geohash_te"].fillna(global_mean)
    train_df["geo_hour_demand"] = train_df["geo_hour_demand"].fillna(train_df["geohash_te"])
    train_df["geo_weather_demand"] = train_df["geo_weather_demand"].fillna(train_df["geohash_te"])
    train_df["loc_te"]     = train_df["loc_te"].fillna(train_df["geohash_te"])
    train_df["loc_hour_te"] = train_df["loc_hour_te"].fillna(train_df["loc_te"])
    train_df["loc_weather_te"] = train_df["loc_weather_te"].fillna(train_df["loc_te"])

    # Compute full training set stats to apply to test_df
    geo_stats = train_df.groupby("geohash")["demand"].agg(["mean", "count"])
    geo_te = ((geo_stats["count"] * geo_stats["mean"] + m1 * global_mean) / (geo_stats["count"] + m1)).to_dict()
    test_df["geohash_te"] = test_df["geohash"].map(geo_te).fillna(global_mean)

    gh_stats = train_df.groupby(["geohash", "hour"])["demand"].mean().reset_index(name="val")
    test_merged = test_df[["geohash", "hour"]].merge(gh_stats, on=["geohash", "hour"], how="left")
    test_df["geo_hour_demand"] = test_merged["val"].fillna(test_df["geohash_te"]).values

    gw_stats = train_df.groupby(["geohash", "Weather"])["demand"].mean().reset_index(name="val")
    test_merged = test_df[["geohash", "Weather"]].merge(gw_stats, on=["geohash", "Weather"], how="left")
    test_df["geo_weather_demand"] = test_merged["val"].fillna(test_df["geohash_te"]).values

    loc_stats = train_df.groupby("loc_key")["demand"].agg(["mean", "count"])
    loc_to_geo = train_df.groupby("loc_key")["geohash"].first().to_dict()
    loc_stats["geo_te"] = loc_stats.index.map(loc_to_geo).map(geo_te).fillna(global_mean)
    loc_te = ((loc_stats["count"] * loc_stats["mean"] + m2 * loc_stats["geo_te"]) / (loc_stats["count"] + m2)).to_dict()
    test_df["loc_te"] = test_df["loc_key"].map(loc_te).fillna(test_df["geohash_te"])

    lh_stats = train_df.groupby(["loc_key", "hour"])["demand"].agg(["mean", "count"]).reset_index()
    lh_stats["loc_te"] = lh_stats["loc_key"].map(loc_te).fillna(global_mean)
    lh_stats["te"] = (lh_stats["count"] * lh_stats["mean"] + m3 * lh_stats["loc_te"]) / (lh_stats["count"] + m3)
    test_merged = test_df[["loc_key", "hour"]].merge(lh_stats[["loc_key", "hour", "te"]], on=["loc_key", "hour"], how="left")
    test_df["loc_hour_te"] = test_merged["te"].fillna(test_df["loc_te"]).values

    lw_stats = train_df.groupby(["loc_key", "Weather"])["demand"].agg(["mean", "count"]).reset_index()
    lw_stats["loc_te"] = lw_stats["loc_key"].map(loc_te).fillna(global_mean)
    lw_stats["te"] = (lw_stats["count"] * lw_stats["mean"] + m4 * lw_stats["loc_te"]) / (lw_stats["count"] + m4)
    test_merged = test_df[["loc_key", "Weather"]].merge(lw_stats[["loc_key", "Weather", "te"]], on=["loc_key", "Weather"], how="left")
    test_df["loc_weather_te"] = test_merged["te"].fillna(test_df["loc_te"]).values

    # Geohash prefix frequency
    for plen in (3, 4):
        col  = f"_gp{plen}"
        fcol = f"geo_prefix{plen}_freq"
        for df in (train_df, test_df):
            df[col] = df["geohash"].str[:plen]
        freq = train_df[col].value_counts().to_dict()
        train_df[fcol] = train_df[col].map(freq)
        test_df[fcol]  = test_df[col].map(freq).fillna(0)
        train_df.drop(columns=[col], inplace=True)
        test_df.drop(columns=[col], inplace=True)

    return train_df, test_df

def add_interaction_features(df):
    df["road_lanes"]   = df["RoadType"].astype(str) + "_" + df["NumberofLanes"].astype(str)
    df["weather_hour"] = df["Weather"].astype(str) + "_" + df["hour"].astype(str)
    df["geo1_hour"]    = df["geo_char_1"].astype(str) + "_" + df["hour"].astype(str)
    df["landmark_road"] = df["Landmarks"].astype(str) + "_" + df["RoadType"].astype(str)
    df["weather_temp"] = df["Weather"].astype(str) + "_" + df["temp_bin"].astype(str)
    df["geo_weather"]  = df["geo_char_1"].astype(str) + "_" + df["Weather"].astype(str)
    return df

_CAT_COLS = [
    "RoadType", "LargeVehicles", "Landmarks", "Weather",
    "geo_char_1", "geo_char_2", "geo_char_3",
    "geo_char_4", "geo_char_5", "geo_char_6",
    "road_lanes", "weather_hour", "geo1_hour",
    "landmark_road", "weather_temp", "geo_weather",
    "loc_key"
]

def encode_categoricals(train_df, test_df):
    for col in _CAT_COLS:
        if col not in train_df.columns:
            continue
        le = LabelEncoder()
        train_df[col] = train_df[col].astype(str).fillna("_na_")
        test_df[col]  = test_df[col].astype(str).fillna("_na_")
        combined = pd.concat([train_df[col], test_df[col]])
        le.fit(combined)
        train_df[col] = le.transform(train_df[col])
        test_df[col]  = le.transform(test_df[col])
    return train_df, test_df

FEATURE_DROP = {"Index", "demand", "geohash", "timestamp", "day"}

def get_feature_cols(df):
    return [c for c in df.columns if c not in FEATURE_DROP]

def engineer_all_features(train_df, test_df):
    train_df = clean_raw(train_df)
    test_df  = clean_raw(test_df)
    train_df = add_temporal_features(train_df)
    test_df  = add_temporal_features(test_df)
    train_df = add_weather_features(train_df)
    test_df  = add_weather_features(test_df)
    train_df, test_df = add_spatial_temporal_features(train_df, test_df)
    train_df = add_interaction_features(train_df)
    test_df  = add_interaction_features(test_df)
    train_df, test_df = encode_categoricals(train_df, test_df)
    for df in (train_df, test_df):
        df.fillna(0, inplace=True)
    return train_df, test_df

# Load raw data
train_raw = pd.read_csv("data/train.csv")
test_raw  = pd.read_csv("data/test.csv")

# Engineer features
train_df, test_df = engineer_all_features(train_raw, test_raw)
feature_cols = get_feature_cols(train_df)

# Validation split
train_df_sorted = train_df.sort_values("minute_of_day").reset_index(drop=True)
VAL_RATIO = 0.20
split_idx = int(len(train_df_sorted) * (1 - VAL_RATIO))
df_tr  = train_df_sorted.iloc[:split_idx]
df_val = train_df_sorted.iloc[split_idx:]

X_tr,  y_tr  = df_tr[feature_cols],  df_tr["demand"]
X_val, y_val = df_val[feature_cols], df_val["demand"]

# Train baseline models
print("Training models with combined geohash and loc-key target encoding...")
from catboost import CatBoostRegressor
import lightgbm as lgb
import xgboost as xgb

cb = CatBoostRegressor(iterations=3000, depth=8, learning_rate=0.05, l2_leaf_reg=3, random_seed=42, verbose=0, early_stopping_rounds=300)
cb.fit(X_tr, y_tr, eval_set=(X_val, y_val))
p_cb = cb.predict(X_val)
print(f"CatBoost val R2: {r2_score(y_val, p_cb):.6f}")

lgb_m = lgb.LGBMRegressor(objective="regression", metric="rmse", num_leaves=63, max_depth=8, learning_rate=0.05, n_estimators=3000, random_state=42, verbose=-1)
lgb_m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(300, verbose=False)])
p_lgb = lgb_m.predict(X_val)
print(f"LightGBM val R2: {r2_score(y_val, p_lgb):.6f}")

xgb_m = xgb.XGBRegressor(objective="reg:squarederror", max_depth=8, learning_rate=0.05, n_estimators=3000, random_state=42, verbosity=0, early_stopping_rounds=300)
xgb_m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
p_xgb = xgb_m.predict(X_val)
print(f"XGBoost val R2: {r2_score(y_val, p_xgb):.6f}")

# Blend
from scipy.optimize import minimize as sp_min
preds_list = [p_cb, p_lgb, p_xgb]
def neg_r2(w):
    w = np.abs(w)
    w = w / w.sum()
    blend = sum(w[i] * preds_list[i] for i in range(3))
    return -r2_score(y_true=y_val, y_pred=blend)

res = sp_min(neg_r2, [0.4, 0.4, 0.2], method="Nelder-Mead")
w = np.abs(res.x)
w = w / w.sum()
blend_val = sum(w[i] * preds_list[i] for i in range(3))
print(f"Ensemble val R2: {r2_score(y_val, blend_val):.6f} with weights CB={w[0]:.3f} LGB={w[1]:.3f} XGB={w[2]:.3f}")
