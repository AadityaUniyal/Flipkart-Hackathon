import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
import warnings

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════
# Phase 0 – General Data Cleaning
# ═══════════════════════════════════════════════════════════════════

def clean_raw(df):
    """Fill obvious missing values in raw columns."""
    df["RoadType"]      = df["RoadType"].fillna("Unknown")
    df["Weather"]       = df["Weather"].fillna("Unknown")
    df["LargeVehicles"] = df["LargeVehicles"].fillna("Unknown")
    df["Landmarks"]     = df["Landmarks"].fillna("Unknown")
    df["NumberofLanes"]  = df["NumberofLanes"].fillna(
        df["NumberofLanes"].median()
    )
    return df


# ═══════════════════════════════════════════════════════════════════
# Phase 3 – Temporal Features
# ═══════════════════════════════════════════════════════════════════

def add_temporal_features(df):
    """Parse *timestamp* (``H:M``) and derive temporal signals."""
    parts = df["timestamp"].str.split(":", expand=True).astype(int)
    df["hour"]          = parts[0]
    df["minute"]        = parts[1]
    df["minute_of_day"] = df["hour"] * 60 + df["minute"]

    # Part of day  0=night  1=morning  2=afternoon  3=evening
    df["part_of_day"] = np.select(
        [df["hour"].between(0, 5),
         df["hour"].between(6, 11),
         df["hour"].between(12, 17),
         df["hour"].between(18, 23)],
        [0, 1, 2, 3], default=0,
    )

    # Peak / rush hour flags
    df["is_peak_hour"] = df["hour"].isin([7, 8, 9, 17, 18, 19]).astype(int)
    df["is_rush_hour"] = df["hour"].isin(
        list(range(6, 11)) + list(range(16, 21))
    ).astype(int)

    # Cyclic encodings
    df["sin_hour"]  = np.sin(2 * np.pi * df["hour"] / 24)
    df["cos_hour"]  = np.cos(2 * np.pi * df["hour"] / 24)
    df["sin_min"]   = np.sin(2 * np.pi * df["minute"] / 60)
    df["cos_min"]   = np.cos(2 * np.pi * df["minute"] / 60)
    df["sin_mod"]   = np.sin(2 * np.pi * df["minute_of_day"] / 1440)
    df["cos_mod"]   = np.cos(2 * np.pi * df["minute_of_day"] / 1440)
    df["day_mod7"]  = df["day"] % 7
    df["sin_day"]   = np.sin(2 * np.pi * df["day_mod7"] / 7)
    df["cos_day"]   = np.cos(2 * np.pi * df["day_mod7"] / 7)
    df["is_weekend"] = df["day_mod7"].isin([5, 6]).astype(int)

    return df


# ═══════════════════════════════════════════════════════════════════
# Phase 5 – Weather & Temperature  (run before geohash aggs)
# ═══════════════════════════════════════════════════════════════════

def add_weather_features(df):
    """Clean weather labels, impute temperature, create bins."""
    # Day / night
    df["is_day"] = ((df["hour"] >= 6) & (df["hour"] < 18)).astype(int)

    # Weather anomaly – "Sunny" at night
    df["weather_anomaly"] = (
        (df["Weather"] == "Sunny") & (df["is_day"] == 0)
    ).astype(int)

    # Temperature imputation: median by (Weather), then global
    weather_med = df.groupby("Weather")["Temperature"].transform("median")
    df["Temperature"] = df["Temperature"].fillna(weather_med)
    df["Temperature"] = df["Temperature"].fillna(df["Temperature"].median())

    # Temperature bins: 0 = cold (<15), 1 = normal (15-30), 2 = hot (>30)
    df["temp_bin"] = np.select(
        [df["Temperature"] < 15,
         df["Temperature"].between(15, 30),
         df["Temperature"] > 30],
        [0, 1, 2], default=1,
    )

    return df


# ═══════════════════════════════════════════════════════════════════
# Phase 4 – Geohash Features  (needs *hour* and *Weather*)
# ═══════════════════════════════════════════════════════════════════

def decode_geohash(geohash):
    base32 = "0123456789bcdefghjkmnpqrstuvwxyz"
    char_map = {char: i for i, char in enumerate(base32)}
    lat_interval = (-90.0, 90.0)
    lon_interval = (-180.0, 180.0)
    is_even = True
    for char in geohash:
        if char not in char_map:
            return 0.0, 0.0
        val = char_map[char]
        for mask in [16, 8, 4, 2, 1]:
            bit = 1 if (val & mask) else 0
            if is_even:
                mid = (lon_interval[0] + lon_interval[1]) / 2
                if bit:
                    lon_interval = (mid, lon_interval[1])
                else:
                    lon_interval = (lon_interval[0], mid)
            else:
                mid = (lat_interval[0] + lat_interval[1]) / 2
                if bit:
                    lat_interval = (mid, lat_interval[1])
                else:
                    lat_interval = (lat_interval[0], mid)
            is_even = not is_even
    lat = (lat_interval[0] + lat_interval[1]) / 2
    lon = (lon_interval[0] + lon_interval[1]) / 2
    return lat, lon


def add_geohash_features(train_df, test_df):
    """Character splits, freq / target encoding, spatial aggregations, clustering, and location profiling."""
    global_mean = train_df["demand"].mean()
    m = 20

    # ── Decode geohash to lat/lon ──────────────────────────────────
    for df in (train_df, test_df):
        coords = df["geohash"].apply(decode_geohash)
        df["latitude"]  = [c[0] for c in coords]
        df["longitude"] = [c[1] for c in coords]

    # ── KMeans Spatial Clustering ──────────────────────────────────
    unique_coords = train_df[["latitude", "longitude"]].drop_duplicates()
    kmeans = KMeans(n_clusters=30, random_state=42, n_init=10)
    kmeans.fit(unique_coords)
    for df in (train_df, test_df):
        df["geo_cluster"] = kmeans.predict(df[["latitude", "longitude"]])

    # ── Define loc_key ─────────────────────────────────────────────
    for df in (train_df, test_df):
        df["loc_key"] = (
            df["geohash"].astype(str) + "_" +
            df["RoadType"].astype(str) + "_" +
            df["NumberofLanes"].astype(str) + "_" +
            df["LargeVehicles"].astype(str) + "_" +
            df["Landmarks"].astype(str)
        )

    # ── Location Profiling (Baseline Day stats) ───────────────────
    min_day = train_df["day"].min()
    baseline_df = train_df[train_df["day"] == min_day]
    loc_stats = baseline_df.groupby("loc_key")["demand"].agg(
        loc_demand_std="std",
        loc_demand_p90=lambda x: np.percentile(x, 90),
        loc_demand_p10=lambda x: np.percentile(x, 10),
        loc_demand_sum="sum"
    ).reset_index()

    train_df["orig_idx"] = range(len(train_df))
    train_df = train_df.merge(loc_stats, on="loc_key", how="left").sort_values("orig_idx").drop(columns=["orig_idx"]).reset_index(drop=True)

    test_df["orig_idx"] = range(len(test_df))
    test_df = test_df.merge(loc_stats, on="loc_key", how="left").sort_values("orig_idx").drop(columns=["orig_idx"]).reset_index(drop=True)

    for df in (train_df, test_df):
        df["loc_demand_std"] = df["loc_demand_std"].fillna(0.0)
        df["loc_demand_p90"] = df["loc_demand_p90"].fillna(global_mean)
        df["loc_demand_p10"] = df["loc_demand_p10"].fillna(global_mean)
        df["loc_demand_sum"] = df["loc_demand_sum"].fillna(0.0)

    # ── Character splitting ────────────────────────────────────────
    max_len = min(6, train_df["geohash"].str.len().min())
    for df in (train_df, test_df):
        for i in range(max_len):
            df[f"geo_char_{i+1}"] = df["geohash"].str[i]

    # ── Frequency encoding ─────────────────────────────────────────
    geo_freq = train_df["geohash"].value_counts().to_dict()
    train_df["geohash_freq"] = train_df["geohash"].map(geo_freq)
    test_df["geohash_freq"]  = test_df["geohash"].map(geo_freq).fillna(0)

    # Initialize target encoding columns
    for df in (train_df, test_df):
        for col in ["geohash_te", "geo_hour_demand", "geo_weather_demand"]:
            df[col] = np.nan

    # ── OOF Target encoding for train_df (5-fold) ───────────────────
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for tr_idx, val_idx in kf.split(train_df):
        tr_fold = train_df.iloc[tr_idx]

        # geohash_te
        stats = tr_fold.groupby("geohash")["demand"].agg(["mean", "count"])
        te_val = (stats["count"] * stats["mean"] + m * global_mean) / (stats["count"] + m)
        train_df.iloc[val_idx, train_df.columns.get_loc("geohash_te")] = train_df.iloc[val_idx]["geohash"].map(te_val)

        # geo_hour_demand
        stats_hour = tr_fold.groupby(["geohash", "hour"])["demand"].mean().reset_index(name="val")
        val_slice = train_df.iloc[val_idx][["geohash", "hour"]]
        val_merged = val_slice.merge(stats_hour, on=["geohash", "hour"], how="left")
        val_merged["val"] = val_merged["val"].fillna(pd.Series(train_df.iloc[val_idx]["geohash"].map(te_val).fillna(global_mean).values, index=val_merged.index))
        train_df.iloc[val_idx, train_df.columns.get_loc("geo_hour_demand")] = val_merged["val"].values

        # geo_weather_demand
        stats_weather = tr_fold.groupby(["geohash", "Weather"])["demand"].mean().reset_index(name="val")
        val_slice = train_df.iloc[val_idx][["geohash", "Weather"]]
        val_merged = val_slice.merge(stats_weather, on=["geohash", "Weather"], how="left")
        val_merged["val"] = val_merged["val"].fillna(pd.Series(train_df.iloc[val_idx]["geohash"].map(te_val).fillna(global_mean).values, index=val_merged.index))
        train_df.iloc[val_idx, train_df.columns.get_loc("geo_weather_demand")] = val_merged["val"].values

    # Fill residual NaNs in train_df
    train_df["geohash_te"] = train_df["geohash_te"].fillna(global_mean)
    train_df["geo_hour_demand"] = train_df["geo_hour_demand"].fillna(train_df["geohash_te"])
    train_df["geo_weather_demand"] = train_df["geo_weather_demand"].fillna(train_df["geohash_te"])

    # ── Test encoding (using full train_df stats) ──────────────────
    stats_full = train_df.groupby("geohash")["demand"].agg(["mean", "count"])
    te_map = ((stats_full["count"] * stats_full["mean"] + m * global_mean) / (stats_full["count"] + m)).to_dict()
    test_df["geohash_te"] = test_df["geohash"].map(te_map).fillna(global_mean)

    stats_hour = train_df.groupby(["geohash", "hour"])["demand"].mean().reset_index(name="val")
    test_merged = test_df[["geohash", "hour"]].merge(stats_hour, on=["geohash", "hour"], how="left")
    test_df["geo_hour_demand"] = test_merged["val"].fillna(test_df["geohash_te"]).values

    stats_weather = train_df.groupby(["geohash", "Weather"])["demand"].mean().reset_index(name="val")
    test_merged = test_df[["geohash", "Weather"]].merge(stats_weather, on=["geohash", "Weather"], how="left")
    test_df["geo_weather_demand"] = test_merged["val"].fillna(test_df["geohash_te"]).values

    # ── Geohash prefix frequency (3 & 4 chars) ────────────────────
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


def add_historical_demand_features(train_df, test_df):
    """Add leakage-safe demand features from strictly earlier days."""
    global_mean = train_df["demand"].mean()
    keys = ["geohash", "day", "minute_of_day"]
    history = train_df[keys + ["demand"]].copy()

    # Shift observations forward one day so every match comes from day-1.
    history["day"] += 1
    history = history.rename(columns={"demand": "demand_prev_day"})

    for lag in (1, 2, 4):
        lagged = history[keys + ["demand_prev_day"]].copy()
        lagged["minute_of_day"] += lag * 15
        lagged = lagged.rename(
            columns={"demand_prev_day": f"demand_lag{lag}"}
        )
        history = history.merge(lagged, on=keys, how="left")

    history["demand_roll4"] = history[
        ["demand_lag1", "demand_lag2", "demand_lag4"]
    ].mean(axis=1)

    feature_cols = [
        "demand_prev_day", "demand_lag1", "demand_lag2",
        "demand_lag4", "demand_roll4",
    ]
    train_df = train_df.merge(history[keys + feature_cols], on=keys, how="left")
    test_df = test_df.merge(history[keys + feature_cols], on=keys, how="left")

    geo_mean = train_df.groupby("geohash")["demand"].mean()
    for df in (train_df, test_df):
        had_history = df["demand_prev_day"].notna()
        fallback = df["geohash"].map(geo_mean).fillna(global_mean)
        for col in feature_cols:
            df[col] = df[col].fillna(fallback)
        df["has_prev_day_demand"] = had_history.astype(int)

    return train_df, test_df


def add_geo_ts_mean_demand_feature(train_df, test_df):
    """Add true geo_ts_mean_demand (mean demand per geohash+timestamp across days)
    using a leave-one-day-out calculation on train to prevent target leakage."""
    global_mean = train_df["demand"].mean()

    # Calculate sum and count of demand for each (geohash, timestamp) in train_df
    geo_ts_stats = train_df.groupby(["geohash", "timestamp"])["demand"].agg(["sum", "count"]).reset_index()

    # Merge stats back to train_df
    train_df = train_df.merge(geo_ts_stats, on=["geohash", "timestamp"], how="left")

    # Leave-one-day-out calculation: subtract current row's demand from sum, and 1 from count
    train_df["geo_ts_mean_demand"] = (train_df["sum"] - train_df["demand"]) / (train_df["count"] - 1)

    # If count is 1, then count - 1 is 0, which results in NaN. Fallback to geohash target encoding or global mean
    train_df["geo_ts_mean_demand"] = train_df["geo_ts_mean_demand"].fillna(train_df["geohash_te"])

    # Drop the temporary sum and count columns
    train_df.drop(columns=["sum", "count"], inplace=True)

    # For test_df, use the overall mean across all training days
    geo_ts_mean = train_df.groupby(["geohash", "timestamp"])["demand"].mean().reset_index(name="geo_ts_mean_demand")
    test_df = test_df.merge(geo_ts_mean, on=["geohash", "timestamp"], how="left")
    test_df["geo_ts_mean_demand"] = test_df["geo_ts_mean_demand"].fillna(test_df["geohash_te"])

    return train_df, test_df



def add_early_morning_features(train_df, test_df):
    """Compute zero-imputed early morning (0:00 to 2:00) stats, activity ratios, neighbor propagation, late morning trends, and temporal decay dynamically."""
    def to_mins(t):
        h, m = map(int, t.split(":"))
        return h * 60 + m

    train_cp = train_df.copy()
    train_cp["minutes"] = train_cp["timestamp"].apply(to_mins)

    early_df = train_cp[train_cp["minutes"] <= 120]

    all_geos = sorted(set(train_df["geohash"].unique()).union(set(test_df["geohash"].unique())))
    early_minutes = [0, 15, 30, 45, 60, 75, 90, 105, 120]

    # Spatial Neighbors Computation (Euclidean distance on coordinates)
    from sklearn.neighbors import BallTree
    unique_geos = list(all_geos)
    geo_coords = {g: decode_geohash(g) for g in unique_geos}
    coords_arr = np.array([geo_coords[g] for g in unique_geos])
    coords_rad = np.radians(coords_arr)
    tree = BallTree(coords_rad, metric="haversine")
    _, neighbor_indices = tree.query(
        coords_rad, k=min(9, len(unique_geos))
    )
    neighbor_indices = neighbor_indices[:, 1:]
    neighbors_dict = {unique_geos[i]: [unique_geos[idx] for idx in neighbor_indices[i]] for i in range(len(unique_geos))}

    # Process each day dynamically
    unique_days = sorted(early_df["day"].unique())
    stats_list = []

    for d in unique_days:
        grid_d = pd.MultiIndex.from_product([all_geos, early_minutes], names=["geohash", "minutes"]).to_frame().reset_index(drop=True)
        early_d = early_df[early_df["day"] == d][["geohash", "minutes", "demand"]]
        grid_d = pd.merge(grid_d, early_d, on=["geohash", "minutes"], how="left").fillna(0.0)
        
        # Compute baseline stats
        stats_d = grid_d.groupby("geohash")["demand"].agg(
            early_mean="mean",
            early_std="std",
            early_max="max",
            early_sum="sum",
            early_last=lambda x: x.iloc[-1]
        ).reset_index()
        stats_d["day"] = d

        # Compute late morning stats (1:00 AM to 2:00 AM)
        late_d = grid_d[grid_d["minutes"] >= 60]
        stats_late_d = late_d.groupby("geohash")["demand"].agg(
            early_late_mean="mean",
            early_late_max="max",
            early_late_std="std",
            early_slope=lambda x: x.iloc[-1] - x.iloc[0]
        ).reset_index()
        stats_d = pd.merge(stats_d, stats_late_d, on="geohash", how="left")

        # Neighbor propagation
        stats_indexed = stats_d.set_index("geohash")
        neighbor_early_mean = []
        neighbor_early_max = []
        neighbor_early_last = []
        for g in stats_d["geohash"]:
            ns = neighbors_dict[g]
            neighbor_early_mean.append(np.mean(stats_indexed.loc[ns, "early_mean"].values))
            neighbor_early_max.append(np.mean(stats_indexed.loc[ns, "early_max"].values))
            neighbor_early_last.append(np.mean(stats_indexed.loc[ns, "early_last"].values))
        stats_d["neighbor_early_mean"] = neighbor_early_mean
        stats_d["neighbor_early_max"] = neighbor_early_max
        stats_d["neighbor_early_last"] = neighbor_early_last

        stats_list.append(stats_d)

    if len(stats_list) > 0:
        stats_all = pd.concat(stats_list, ignore_index=True)
    else:
        # Fallback if no early morning data at all
        stats_all = pd.DataFrame(columns=["geohash", "day", "early_mean", "early_std", "early_max", "early_sum", "early_last",
                                          "early_late_mean", "early_late_max", "early_late_std", "early_slope",
                                          "neighbor_early_mean", "neighbor_early_max", "neighbor_early_last"])

    # Merge
    train_df = pd.merge(train_df, stats_all, on=["geohash", "day"], how="left")
    test_df = pd.merge(test_df, stats_all, on=["geohash", "day"], how="left")

    for df in (train_df, test_df):
        df["early_mean"] = df["early_mean"].fillna(0.0)
        df["early_std"] = df["early_std"].fillna(0.0)
        df["early_max"] = df["early_max"].fillna(0.0)
        df["early_sum"] = df["early_sum"].fillna(0.0)
        df["early_last"] = df["early_last"].fillna(0.0)

        df["neighbor_early_mean"] = df["neighbor_early_mean"].fillna(0.0)
        df["neighbor_early_max"] = df["neighbor_early_max"].fillna(0.0)
        df["neighbor_early_last"] = df["neighbor_early_last"].fillna(0.0)

        df["early_late_mean"] = df["early_late_mean"].fillna(0.0)
        df["early_late_max"] = df["early_late_max"].fillna(0.0)
        df["early_late_std"] = df["early_late_std"].fillna(0.0)
        df["early_slope"] = df["early_slope"].fillna(0.0)

        # Activity ratios against target encoding (geohash_te)
        df["early_ratio_mean"] = df["early_mean"] / (df["geohash_te"] + 1e-5)
        df["early_ratio_max"] = df["early_max"] / (df["geohash_te"] + 1e-5)

        # Temporal Decay Feature
        time_elapsed = np.maximum(0, df["minute_of_day"] - 120)
        df["early_decay"] = np.exp(-time_elapsed / 120.0)
        
        # Add decayed early morning statistics
        df["early_mean_decayed"] = df["early_mean"] * df["early_decay"]
        df["early_max_decayed"] = df["early_max"] * df["early_decay"]
        df["early_last_decayed"] = df["early_last"] * df["early_decay"]

    return train_df, test_df





# ═══════════════════════════════════════════════════════════════════
# Phase 6 – Interaction Features
# ═══════════════════════════════════════════════════════════════════

def add_interaction_features(df):
    """Categorical interaction columns."""
    df["road_lanes"]   = df["RoadType"].astype(str) + "_" + df["NumberofLanes"].astype(str)
    df["weather_hour"] = df["Weather"].astype(str) + "_" + df["hour"].astype(str)
    df["geo1_hour"]    = df["geo_char_1"].astype(str) + "_" + df["hour"].astype(str)
    df["landmark_road"] = df["Landmarks"].astype(str) + "_" + df["RoadType"].astype(str)
    df["weather_temp"] = df["Weather"].astype(str) + "_" + df["temp_bin"].astype(str)
    df["geo_weather"]  = df["geo_char_1"].astype(str) + "_" + df["Weather"].astype(str)
    df["vehicle_road"] = df["LargeVehicles"].astype(str) + "_" + df["RoadType"].astype(str)
    df["vehicle_lanes"] = df["LargeVehicles"].astype(str) + "_" + df["NumberofLanes"].astype(str)
    df["road_lanes_vehicles"] = df["RoadType"].astype(str) + "_" + df["NumberofLanes"].astype(str) + "_" + df["LargeVehicles"].astype(str)
    return df


# ═══════════════════════════════════════════════════════════════════
# Label Encoding
# ═══════════════════════════════════════════════════════════════════

_CAT_COLS = [
    "RoadType", "LargeVehicles", "Landmarks", "Weather",
    "geo_char_1", "geo_char_2", "geo_char_3",
    "geo_char_4", "geo_char_5", "geo_char_6",
    "road_lanes", "weather_hour", "geo1_hour",
    "landmark_road", "weather_temp", "geo_weather",
    "vehicle_road", "vehicle_lanes", "road_lanes_vehicles",
    "loc_key", "geo_cluster",
]


def encode_categoricals(train_df, test_df):
    """Label-encode string columns so tree models can consume them."""
    encoders = {}
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
        encoders[col] = le
    return train_df, test_df, encoders


# ═══════════════════════════════════════════════════════════════════
# Master entry-point
# ═══════════════════════════════════════════════════════════════════

FEATURE_DROP = {"Index", "demand", "geohash", "timestamp", "day"}


def get_feature_cols(df):
    """Return list of feature column names."""
    return [c for c in df.columns if c not in FEATURE_DROP]


def engineer_all_features(train_df, test_df):
    """Run the complete feature-engineering pipeline.

    Parameters
    ----------
    train_df, test_df : raw DataFrames straight from CSV.

    Returns
    -------
    train_df, test_df : DataFrames with all features added.
    """
    print("  [0/6] Cleaning raw data ...")
    train_df = clean_raw(train_df)
    test_df  = clean_raw(test_df)

    print("  [1/6] Temporal features ...")
    train_df = add_temporal_features(train_df)
    test_df  = add_temporal_features(test_df)

    print("  [2/6] Weather & temperature ...")
    train_df = add_weather_features(train_df)
    test_df  = add_weather_features(test_df)

    # Compute hourly median temp on train_df and compute temp_dev
    hourly_med_temp = train_df.groupby("hour")["Temperature"].median().to_dict()
    train_df["temp_dev"] = train_df["Temperature"] - train_df["hour"].map(hourly_med_temp)
    test_df["temp_dev"]  = test_df["Temperature"] - test_df["hour"].map(hourly_med_temp).fillna(train_df["Temperature"].median())

    print("  [3/6] Geohash features ...")
    train_df, test_df = add_geohash_features(train_df, test_df)
    train_df, test_df = add_historical_demand_features(train_df, test_df)
    train_df, test_df = add_geo_ts_mean_demand_feature(train_df, test_df)
    train_df, test_df = add_early_morning_features(train_df, test_df)

    print("  [4/6] Interaction features ...")
    train_df = add_interaction_features(train_df)
    test_df  = add_interaction_features(test_df)

    print("  [5/6] Encoding categoricals ...")
    train_df, test_df, _ = encode_categoricals(train_df, test_df)

    # Add continuous cyclic interactions
    for df in (train_df, test_df):
        df["sin_hour_road"] = df["sin_hour"] * df["RoadType"]
        df["cos_hour_road"] = df["cos_hour"] * df["RoadType"]

    print("  [6/6] Filling residual NaN ...")
    for df in (train_df, test_df):
        df.fillna(0, inplace=True)
        for col in df.select_dtypes(include=["float64"]).columns:
            df[col] = df[col].astype(np.float32)
        for col in df.select_dtypes(include=["int64"]).columns:
            df[col] = df[col].astype(np.int32)

    feat_cols = get_feature_cols(train_df)
    print(f"  * Done - {len(feat_cols)} features")
    return train_df, test_df