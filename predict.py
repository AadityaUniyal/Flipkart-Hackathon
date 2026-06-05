import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
from sklearn.neighbors import BallTree

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
    df["NumberofLanes"] = df["NumberofLanes"].fillna(
        df["NumberofLanes"].median()
    )
    return df


# ═══════════════════════════════════════════════════════════════════
# Phase 1 – Temporal Features
# ═══════════════════════════════════════════════════════════════════

def add_temporal_features(df):
    """Parse timestamp (H:M) and derive all temporal signals."""
    parts = df["timestamp"].str.split(":", expand=True).astype(int)
    df["hour"]          = parts[0]
    df["minute"]        = parts[1]
    df["minute_of_day"] = df["hour"] * 60 + df["minute"]

    # Part of day: 0=night  1=morning  2=afternoon  3=evening
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

    # Cyclic encodings – hour, minute, minute_of_day
    df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["sin_min"]  = np.sin(2 * np.pi * df["minute"] / 60)
    df["cos_min"]  = np.cos(2 * np.pi * df["minute"] / 60)
    df["sin_mod"]  = np.sin(2 * np.pi * df["minute_of_day"] / 1440)
    df["cos_mod"]  = np.cos(2 * np.pi * df["minute_of_day"] / 1440)

    # Day-of-week cyclicity
    df["day_mod7"]   = df["day"] % 7
    df["sin_day"]    = np.sin(2 * np.pi * df["day_mod7"] / 7)
    df["cos_day"]    = np.cos(2 * np.pi * df["day_mod7"] / 7)
    df["is_weekend"] = df["day_mod7"].isin([5, 6]).astype(int)

    return df


# ═══════════════════════════════════════════════════════════════════
# Phase 2 – Weather & Temperature Features
# ═══════════════════════════════════════════════════════════════════

def add_weather_features(df):
    """Clean weather labels, impute temperature, create bins."""
    df["is_day"] = ((df["hour"] >= 6) & (df["hour"] < 18)).astype(int)

    # Weather anomaly – "Sunny" at night
    df["weather_anomaly"] = (
        (df["Weather"] == "Sunny") & (df["is_day"] == 0)
    ).astype(int)

    # Temperature imputation: median by Weather group, then global
    weather_med = df.groupby("Weather")["Temperature"].transform("median")
    df["Temperature"] = df["Temperature"].fillna(weather_med)
    df["Temperature"] = df["Temperature"].fillna(df["Temperature"].median())

    # Temperature bins: 0=cold (<15)  1=normal (15–30)  2=hot (>30)
    df["temp_bin"] = np.select(
        [df["Temperature"] < 15,
         df["Temperature"].between(15, 30),
         df["Temperature"] > 30],
        [0, 1, 2], default=1,
    )

    return df


# ═══════════════════════════════════════════════════════════════════
# Phase 3 – Geohash Features
# ═══════════════════════════════════════════════════════════════════

def decode_geohash(geohash):
    """Decode a geohash string to (lat, lon)."""
    base32 = "0123456789bcdefghjkmnpqrstuvwxyz"
    char_map = {c: i for i, c in enumerate(base32)}
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
                lon_interval = (mid, lon_interval[1]) if bit else (lon_interval[0], mid)
            else:
                mid = (lat_interval[0] + lat_interval[1]) / 2
                lat_interval = (mid, lat_interval[1]) if bit else (lat_interval[0], mid)
            is_even = not is_even
    return (lat_interval[0] + lat_interval[1]) / 2, (lon_interval[0] + lon_interval[1]) / 2


def add_geohash_features(train_df, test_df):
    """
    Decode geohash → lat/lon, KMeans clustering, character splits,
    frequency encoding, OOF target encoding, location profiling.
    """
    global_mean = train_df["demand"].mean()
    m = 20  # smoothing factor for target encoding

    # ── Decode to lat/lon ─────────────────────────────────────────
    for df in (train_df, test_df):
        coords = df["geohash"].apply(decode_geohash)
        df["latitude"]  = [c[0] for c in coords]
        df["longitude"] = [c[1] for c in coords]

    # ── KMeans Spatial Clustering (30 clusters) ───────────────────
    unique_coords = train_df[["latitude", "longitude"]].drop_duplicates()
    kmeans = KMeans(n_clusters=30, random_state=42, n_init=10)
    kmeans.fit(unique_coords)
    for df in (train_df, test_df):
        df["geo_cluster"] = kmeans.predict(df[["latitude", "longitude"]])

    # ── Location key ─────────────────────────────────────────────
    for df in (train_df, test_df):
        df["loc_key"] = (
            df["geohash"].astype(str) + "_" +
            df["RoadType"].astype(str) + "_" +
            df["NumberofLanes"].astype(str) + "_" +
            df["LargeVehicles"].astype(str) + "_" +
            df["Landmarks"].astype(str)
        )

    # ── Location Profiling – use all training days (not just day 48) ──
    loc_stats = train_df.groupby("loc_key")["demand"].agg(
        loc_demand_std="std",
        loc_demand_p90=lambda x: np.percentile(x, 90),
        loc_demand_p10=lambda x: np.percentile(x, 10),
        loc_demand_sum="sum",
    ).reset_index()

    for df in (train_df, test_df):
        orig_idx = df.index.copy()
        df["_orig_idx"] = np.arange(len(df))
        df_merged = df.merge(loc_stats, on="loc_key", how="left").sort_values("_orig_idx").reset_index(drop=True)
        df_merged.drop(columns=["_orig_idx"], inplace=True)
        # Copy merged columns back – safest approach
        for col in ["loc_demand_std", "loc_demand_p90", "loc_demand_p10", "loc_demand_sum"]:
            df[col] = df_merged[col].values

    for df in (train_df, test_df):
        df["loc_demand_std"] = df["loc_demand_std"].fillna(0.0)
        df["loc_demand_p90"] = df["loc_demand_p90"].fillna(global_mean)
        df["loc_demand_p10"] = df["loc_demand_p10"].fillna(global_mean)
        df["loc_demand_sum"] = df["loc_demand_sum"].fillna(0.0)

    # ── Character splitting (up to 6 chars) ──────────────────────
    max_len = min(6, train_df["geohash"].str.len().min())
    for df in (train_df, test_df):
        for i in range(max_len):
            df[f"geo_char_{i+1}"] = df["geohash"].str[i]

    # ── Frequency encoding ────────────────────────────────────────
    geo_freq = train_df["geohash"].value_counts().to_dict()
    train_df["geohash_freq"] = train_df["geohash"].map(geo_freq)
    test_df["geohash_freq"]  = test_df["geohash"].map(geo_freq).fillna(0)

    # ── OOF Target Encoding (5-fold) ─────────────────────────────
    for col in ["geohash_te", "geo_hour_demand", "geo_weather_demand"]:
        train_df[col] = np.nan
        test_df[col]  = np.nan

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for tr_idx, val_idx in kf.split(train_df):
        tr_fold  = train_df.iloc[tr_idx]

        # geohash_te (smoothed mean)
        stats = tr_fold.groupby("geohash")["demand"].agg(["mean", "count"])
        te_val = (stats["count"] * stats["mean"] + m * global_mean) / (stats["count"] + m)

        train_df.iloc[val_idx, train_df.columns.get_loc("geohash_te")] = (
            train_df.iloc[val_idx]["geohash"].map(te_val)
        )

        # geo_hour_demand
        stats_hour = tr_fold.groupby(["geohash", "hour"])["demand"].mean().reset_index(name="val")
        val_slice  = train_df.iloc[val_idx][["geohash", "hour"]].copy()
        val_slice["_vi"] = val_idx
        val_merged = val_slice.merge(stats_hour, on=["geohash", "hour"], how="left")
        fallback   = val_slice["geohash"].map(te_val).fillna(global_mean).values
        filled     = val_merged["val"].fillna(pd.Series(fallback, index=val_merged.index)).values
        train_df.iloc[val_idx, train_df.columns.get_loc("geo_hour_demand")] = filled

        # geo_weather_demand
        stats_weather = tr_fold.groupby(["geohash", "Weather"])["demand"].mean().reset_index(name="val")
        val_slice2    = train_df.iloc[val_idx][["geohash", "Weather"]].copy()
        val_merged2   = val_slice2.merge(stats_weather, on=["geohash", "Weather"], how="left")
        filled2       = val_merged2["val"].fillna(pd.Series(fallback, index=val_merged2.index)).values
        train_df.iloc[val_idx, train_df.columns.get_loc("geo_weather_demand")] = filled2

    # Fill residual NaNs in train
    train_df["geohash_te"]         = train_df["geohash_te"].fillna(global_mean)
    train_df["geo_hour_demand"]    = train_df["geo_hour_demand"].fillna(train_df["geohash_te"])
    train_df["geo_weather_demand"] = train_df["geo_weather_demand"].fillna(train_df["geohash_te"])

    # ── Test encoding (full train stats) ──────────────────────────
    stats_full = train_df.groupby("geohash")["demand"].agg(["mean", "count"])
    te_map = ((stats_full["count"] * stats_full["mean"] + m * global_mean) /
              (stats_full["count"] + m)).to_dict()
    test_df["geohash_te"] = test_df["geohash"].map(te_map).fillna(global_mean)

    stats_hour = train_df.groupby(["geohash", "hour"])["demand"].mean().reset_index(name="val")
    test_merged = test_df[["geohash", "hour"]].merge(stats_hour, on=["geohash", "hour"], how="left")
    test_df["geo_hour_demand"] = test_merged["val"].fillna(test_df["geohash_te"]).values

    stats_weather = train_df.groupby(["geohash", "Weather"])["demand"].mean().reset_index(name="val")
    test_merged2 = test_df[["geohash", "Weather"]].merge(stats_weather, on=["geohash", "Weather"], how="left")
    test_df["geo_weather_demand"] = test_merged2["val"].fillna(test_df["geohash_te"]).values

    # ── Geohash prefix frequency (3 & 4 chars) ───────────────────
    for plen in (3, 4):
        fcol = f"geo_prefix{plen}_freq"
        tmp_col = f"_gp{plen}"
        for df in (train_df, test_df):
            df[tmp_col] = df["geohash"].str[:plen]
        freq = train_df[tmp_col].value_counts().to_dict()
        train_df[fcol] = train_df[tmp_col].map(freq)
        test_df[fcol]  = test_df[tmp_col].map(freq).fillna(0)
        for df in (train_df, test_df):
            df.drop(columns=[tmp_col], inplace=True)

    return train_df, test_df


# ═══════════════════════════════════════════════════════════════════
# Phase 4 – Historical Demand Features (leakage-safe)
# ═══════════════════════════════════════════════════════════════════

def add_historical_demand_features(train_df, test_df):
    """
    For each (geohash, timestamp) row, look up the demand from the
    PREVIOUS day (same geohash, same timestamp, day - 1).

    This is strictly leakage-safe because we only look backwards.
    For test rows, the most recent training day's demand is used.
    """
    global_mean = train_df["demand"].mean()

    # Build a lookup: for day D, fetch demand from day D-1
    prev = train_df[["geohash", "day", "timestamp", "demand"]].copy()
    prev["day"] = prev["day"] + 1          # shift: day D-1 data → day D key
    prev = prev.rename(columns={"demand": "demand_prev_day"})

    # Merge onto train (day D row finds day D-1 demand)
    train_df = train_df.merge(
        prev, on=["geohash", "day", "timestamp"], how="left"
    )
    # Merge onto test (test day D row finds day D-1 demand from train)
    test_df = test_df.merge(
        prev, on=["geohash", "day", "timestamp"], how="left"
    )

    # Fallback: geohash mean, then global mean
    geo_mean = train_df.groupby("geohash")["demand"].mean()

    train_df["has_prev_day_demand"] = train_df["demand_prev_day"].notna().astype(int)
    test_df["has_prev_day_demand"]  = test_df["demand_prev_day"].notna().astype(int)

    fallback_tr = train_df["geohash"].map(geo_mean).fillna(global_mean)
    fallback_te = test_df["geohash"].map(geo_mean).fillna(global_mean)

    train_df["demand_prev_day"] = train_df["demand_prev_day"].fillna(fallback_tr)
    test_df["demand_prev_day"]  = test_df["demand_prev_day"].fillna(fallback_te)

    # ── Same geohash, same timestamp, 7 days ago ─────────────────
    prev7 = train_df[["geohash", "day", "timestamp", "demand"]].copy()
    prev7["day"] = prev7["day"] + 7
    prev7 = prev7.rename(columns={"demand": "demand_prev_week"})

    train_df = train_df.merge(prev7, on=["geohash", "day", "timestamp"], how="left")
    test_df  = test_df.merge(prev7, on=["geohash", "day", "timestamp"], how="left")

    train_df["demand_prev_week"] = train_df["demand_prev_week"].fillna(fallback_tr)
    test_df["demand_prev_week"]  = test_df["demand_prev_week"].fillna(fallback_te)

    # ── Rolling 3-day average (days D-1, D-2, D-3) ──────────────
    for lag in (2, 3):
        prev_lag = train_df[["geohash", "day", "timestamp", "demand"]].copy()
        prev_lag["day"] = prev_lag["day"] + lag
        prev_lag = prev_lag.rename(columns={"demand": f"demand_lag{lag}"})
        train_df = train_df.merge(prev_lag, on=["geohash", "day", "timestamp"], how="left")
        test_df  = test_df.merge(prev_lag, on=["geohash", "day", "timestamp"], how="left")
        train_df[f"demand_lag{lag}"] = train_df[f"demand_lag{lag}"].fillna(fallback_tr)
        test_df[f"demand_lag{lag}"]  = test_df[f"demand_lag{lag}"].fillna(fallback_te)

    train_df["demand_roll3"] = (
        train_df["demand_prev_day"] +
        train_df["demand_lag2"] +
        train_df["demand_lag3"]
    ) / 3.0
    test_df["demand_roll3"] = (
        test_df["demand_prev_day"] +
        test_df["demand_lag2"] +
        test_df["demand_lag3"]
    ) / 3.0

    return train_df, test_df


# ═══════════════════════════════════════════════════════════════════
# Phase 5 – Geo × Timestamp Mean Demand (leave-one-day-out)
# ═══════════════════════════════════════════════════════════════════

def add_geo_ts_mean_demand_feature(train_df, test_df):
    """
    For train: leave-one-day-out mean of demand per (geohash, timestamp).
    For test:  full training mean per (geohash, timestamp).

    This is the single highest-impact feature for periodic traffic data.
    """
    global_mean = train_df["demand"].mean()

    # ── Train: leave-one-day-out ──────────────────────────────────
    # Sum and count across ALL days for each (geohash, timestamp)
    geo_ts_stats = (
        train_df.groupby(["geohash", "timestamp"])["demand"]
        .agg(["sum", "count"])
        .reset_index()
        .rename(columns={"sum": "_gs", "count": "_gc"})
    )
    train_df = train_df.merge(geo_ts_stats, on=["geohash", "timestamp"], how="left")

    # Subtract current row's contribution → unbiased leave-one-out estimate
    denom = (train_df["_gc"] - 1).clip(lower=1)
    train_df["geo_ts_mean_demand"] = (train_df["_gs"] - train_df["demand"]) / denom
    # Where count was 1, the above is NaN → fall back to geohash target encoding
    train_df["geo_ts_mean_demand"] = (
        train_df["geo_ts_mean_demand"]
        .fillna(train_df["geohash_te"])
        .fillna(global_mean)
    )
    train_df.drop(columns=["_gs", "_gc"], inplace=True)

    # ── Test: use full training mean ─────────────────────────────
    geo_ts_mean = (
        train_df.groupby(["geohash", "timestamp"])["demand"]
        .mean()
        .reset_index(name="geo_ts_mean_demand")
    )
    test_df = test_df.merge(geo_ts_mean, on=["geohash", "timestamp"], how="left")
    test_df["geo_ts_mean_demand"] = (
        test_df["geo_ts_mean_demand"]
        .fillna(test_df["geohash_te"])
        .fillna(global_mean)
    )

    # ── Hour-level geo mean (coarser but robust fallback) ────────
    geo_hour_mean = (
        train_df.groupby(["geohash", "hour"])["demand"]
        .mean()
        .reset_index(name="geo_hour_mean")
    )
    for df in (train_df, test_df):
        tmp = df[["geohash", "hour"]].merge(geo_hour_mean, on=["geohash", "hour"], how="left")
        df["geo_hour_mean"] = tmp["geo_hour_mean"].fillna(global_mean).values

    return train_df, test_df


# ═══════════════════════════════════════════════════════════════════
# Phase 6 – Early Morning Features
# ═══════════════════════════════════════════════════════════════════

def add_early_morning_features(train_df, test_df):
    """
    Early morning (0:00–2:00) aggregation stats, spatial neighbor
    propagation via BallTree (haversine), temporal decay.

    FIXED: dynamically detects the last two training days instead of
    hardcoding days 48/49, so this works on any dataset.
    """
    def to_mins(t):
        h, m = map(int, t.split(":"))
        return h * 60 + m

    # ── Detect last two training days dynamically ─────────────────
    sorted_days    = sorted(train_df["day"].unique())
    last_day       = sorted_days[-1]   # was hardcoded as 49
    prev_day_early = sorted_days[-2] if len(sorted_days) >= 2 else sorted_days[-1]

    train_cp = train_df.copy()
    train_cp["minutes"] = train_cp["timestamp"].apply(to_mins)
    early_df = train_cp[train_cp["minutes"] <= 120]

    all_geos     = set(train_df["geohash"].unique()) | set(test_df["geohash"].unique())
    early_minutes = [0, 15, 30, 45, 60, 75, 90, 105, 120]

    def build_early_stats(day_val):
        grid = pd.MultiIndex.from_product(
            [all_geos, early_minutes], names=["geohash", "minutes"]
        ).to_frame().reset_index(drop=True)
        day_data = early_df[early_df["day"] == day_val][["geohash", "minutes", "demand"]]
        grid = grid.merge(day_data, on=["geohash", "minutes"], how="left").fillna(0.0)
        stats = grid.groupby("geohash")["demand"].agg(
            early_mean="mean",
            early_std="std",
            early_max="max",
            early_sum="sum",
            early_last=lambda x: x.iloc[-1],
        ).reset_index()
        stats["day"] = day_val

        # Late morning (1:00–2:00) slope & trend
        late = grid[grid["minutes"] >= 60]
        stats_late = late.groupby("geohash")["demand"].agg(
            early_late_mean="mean",
            early_late_max="max",
            early_late_std="std",
            early_slope=lambda x: x.iloc[-1] - x.iloc[0],
        ).reset_index()
        stats = stats.merge(stats_late, on="geohash", how="left")
        return stats

    stats_prev = build_early_stats(prev_day_early)
    stats_last = build_early_stats(last_day)

    # ── BallTree neighbor propagation ─────────────────────────────
    unique_geos  = list(all_geos)
    geo_coords   = {g: decode_geohash(g) for g in unique_geos}
    coords_arr   = np.array([geo_coords[g] for g in unique_geos])
    coords_rad   = np.radians(coords_arr)
    tree         = BallTree(coords_rad, metric="haversine")
    k_neighbors  = min(9, len(unique_geos))
    _, n_indices = tree.query(coords_rad, k=k_neighbors)
    # Exclude self (index 0 is always the point itself)
    n_indices    = n_indices[:, 1:]
    neighbors_dict = {
        unique_geos[i]: [unique_geos[idx] for idx in n_indices[i]]
        for i in range(len(unique_geos))
    }

    def add_neighbor_stats(stats_df):
        indexed = stats_df.set_index("geohash")
        nbr_mean, nbr_max, nbr_last = [], [], []
        for g in stats_df["geohash"]:
            ns = [n for n in neighbors_dict[g] if n in indexed.index]
            if ns:
                nbr_mean.append(indexed.loc[ns, "early_mean"].mean())
                nbr_max.append(indexed.loc[ns, "early_max"].mean())
                nbr_last.append(indexed.loc[ns, "early_last"].mean())
            else:
                nbr_mean.append(0.0)
                nbr_max.append(0.0)
                nbr_last.append(0.0)
        stats_df["neighbor_early_mean"] = nbr_mean
        stats_df["neighbor_early_max"]  = nbr_max
        stats_df["neighbor_early_last"] = nbr_last
        return stats_df

    stats_prev = add_neighbor_stats(stats_prev)
    stats_last = add_neighbor_stats(stats_last)

    # ── Merge into train (each row gets stats from its own day) ──
    stats_all = pd.concat([stats_prev, stats_last], ignore_index=True)
    train_df  = train_df.merge(stats_all, on=["geohash", "day"], how="left")

    # ── Merge into test (always use last training day's stats) ───
    stats_last_no_day = stats_last.drop(columns=["day"])
    test_df = test_df.merge(stats_last_no_day, on="geohash", how="left")

    # ── Fill NaNs and compute derived features ────────────────────
    early_cols = [
        "early_mean", "early_std", "early_max", "early_sum", "early_last",
        "neighbor_early_mean", "neighbor_early_max", "neighbor_early_last",
        "early_late_mean", "early_late_max", "early_late_std", "early_slope",
    ]
    for df in (train_df, test_df):
        for col in early_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0.0)

        # Activity ratios vs target encoding
        df["early_ratio_mean"] = df["early_mean"] / (df["geohash_te"] + 1e-5)
        df["early_ratio_max"]  = df["early_max"]  / (df["geohash_te"] + 1e-5)

        # Temporal decay from 2:00 AM onward
        time_elapsed = np.maximum(0, df["minute_of_day"] - 120)
        df["early_decay"]        = np.exp(-time_elapsed / 120.0)
        df["early_mean_decayed"] = df["early_mean"]  * df["early_decay"]
        df["early_max_decayed"]  = df["early_max"]   * df["early_decay"]
        df["early_last_decayed"] = df["early_last"]  * df["early_decay"]

    return train_df, test_df


# ═══════════════════════════════════════════════════════════════════
# Phase 7 – Interaction Features
# ═══════════════════════════════════════════════════════════════════

def add_interaction_features(df):
    """Categorical and numeric interaction columns."""
    df["road_lanes"]          = df["RoadType"].astype(str) + "_" + df["NumberofLanes"].astype(str)
    df["weather_hour"]        = df["Weather"].astype(str) + "_" + df["hour"].astype(str)
    df["geo1_hour"]           = df["geo_char_1"].astype(str) + "_" + df["hour"].astype(str)
    df["landmark_road"]       = df["Landmarks"].astype(str) + "_" + df["RoadType"].astype(str)
    df["weather_temp"]        = df["Weather"].astype(str) + "_" + df["temp_bin"].astype(str)
    df["geo_weather"]         = df["geo_char_1"].astype(str) + "_" + df["Weather"].astype(str)
    df["vehicle_road"]        = df["LargeVehicles"].astype(str) + "_" + df["RoadType"].astype(str)
    df["vehicle_lanes"]       = df["LargeVehicles"].astype(str) + "_" + df["NumberofLanes"].astype(str)
    df["road_lanes_vehicles"] = (
        df["RoadType"].astype(str) + "_" +
        df["NumberofLanes"].astype(str) + "_" +
        df["LargeVehicles"].astype(str)
    )
    return df


# ═══════════════════════════════════════════════════════════════════
# Phase 8 – Label Encoding + Memory Reduction
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
    """Label-encode string columns. Fits on combined train+test vocab."""
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
    """Return list of usable feature column names."""
    return [c for c in df.columns if c not in FEATURE_DROP]


def engineer_all_features(train_df, test_df):
    """
    Run the complete feature-engineering pipeline.

    Parameters
    ----------
    train_df, test_df : raw DataFrames straight from CSV.

    Returns
    -------
    train_df, test_df : DataFrames with all engineered features.
    """
    print("  [0/8] Cleaning raw data ...")
    train_df = clean_raw(train_df)
    test_df  = clean_raw(test_df)

    print("  [1/8] Temporal features ...")
    train_df = add_temporal_features(train_df)
    test_df  = add_temporal_features(test_df)

    print("  [2/8] Weather & temperature ...")
    train_df = add_weather_features(train_df)
    test_df  = add_weather_features(test_df)

    # Temperature deviation from hourly median (computed on train only)
    hourly_med_temp = train_df.groupby("hour")["Temperature"].median().to_dict()
    train_df["temp_dev"] = train_df["Temperature"] - train_df["hour"].map(hourly_med_temp)
    test_df["temp_dev"]  = (
        test_df["Temperature"] -
        test_df["hour"].map(hourly_med_temp).fillna(train_df["Temperature"].median())
    )

    print("  [3/8] Geohash features (OOF target encoding, clustering) ...")
    train_df, test_df = add_geohash_features(train_df, test_df)

    print("  [4/8] Historical demand features (prev-day lag) ...")
    train_df, test_df = add_historical_demand_features(train_df, test_df)

    print("  [5/8] Geo × timestamp mean demand (leave-one-day-out) ...")
    train_df, test_df = add_geo_ts_mean_demand_feature(train_df, test_df)

    print("  [6/8] Early morning features (BallTree neighbors) ...")
    train_df, test_df = add_early_morning_features(train_df, test_df)

    print("  [7/8] Interaction features ...")
    train_df = add_interaction_features(train_df)
    test_df  = add_interaction_features(test_df)

    print("  [8/8] Encoding categoricals + memory reduction ...")
    train_df, test_df, _ = encode_categoricals(train_df, test_df)

    # Cyclic × road-type numeric interactions (post-encoding)
    for df in (train_df, test_df):
        df["sin_hour_road"] = df["sin_hour"] * df["RoadType"]
        df["cos_hour_road"] = df["cos_hour"] * df["RoadType"]

    # Fill any residual NaNs and downcast to save memory
    for df in (train_df, test_df):
        df.fillna(0, inplace=True)
        for col in df.select_dtypes(include=["float64"]).columns:
            df[col] = df[col].astype(np.float32)
        for col in df.select_dtypes(include=["int64"]).columns:
            df[col] = df[col].astype(np.int32)

    feat_cols = get_feature_cols(train_df)
    print(f"  * Done — {len(feat_cols)} features engineered")
    return train_df, test_df
