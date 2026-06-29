"""
preprocess.py
==============
Hydro-AI Sentinel — Kandy Otieno Genga | SCT213-C002-0106/2022
JKUAT BSc. Data Science 2026

PURPOSE:
  Full preprocessing pipeline for the Bi-LSTM flood model:
  1. Load raw CSV  →  linear-interpolation for missing WRA values
  2. Engineer 12 features (rolling stats, lag features, cyclical time)
  3. Min-Max normalise  →  save scaler to disk
  4. Build 30-day sliding-window sequences  (X, y)
  5. Chronological split: 70% train / 15% val / 15% test
  6. Save compressed .npz arrays ready for training

RUN:
  python preprocess.py

OUTPUT:
  models/scaler_nzoia.pkl   models/scaler_tana.pkl
  data/sequences_nzoia.npz  data/sequences_tana.npz
"""

import os
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# ── Configuration ────────────────────────────────────────────────
SEQUENCE_LEN = 30       # days of history in each X window
TRAIN_FRAC   = 0.70
VAL_FRAC     = 0.15
# TEST_FRAC  = 0.15     (remainder)

# Order matters — model will receive features in this order
FEATURE_COLS = [
    "rainfall_mm",      # raw daily satellite rainfall
    "river_level_m",    # ← also the TARGET (index 1)
    "temperature_c",
    "humidity_pct",
    "rain_3d_sum",      # 3-day cumulative rainfall (catchment memory)
    "rain_7d_sum",      # 7-day cumulative rainfall
    "level_lag1",       # river level 1 day ago    (autoregressive)
    "level_lag3",       # river level 3 days ago
    "level_7d_mean",    # 7-day rolling average
    "level_7d_std",     # 7-day rolling std-dev (volatility signal)
    "month_sin",        # cyclical month → captures Long Rains
    "month_cos",        # cyclical month → captures Short Rains
]
TARGET_COL   = "river_level_m"
TARGET_INDEX = FEATURE_COLS.index(TARGET_COL)   # = 1


# ── Step 1: Load & clean ─────────────────────────────────────────
def load_and_clean(basin):
    path = f"data/{basin}_dataset.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run generate_data.py first."
        )

    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    n_missing = df["river_level_m"].isna().sum()

    # Linear interpolation fills WRA data gaps
    df["river_level_m"] = df["river_level_m"].interpolate(
        method="linear", limit_direction="both"
    )
    print(f"  Interpolated {n_missing} missing river-level values")

    # Safety check — no NaN should remain
    assert df["river_level_m"].isna().sum() == 0, "NaN remains after interpolation!"
    return df


# ── Step 2: Feature engineering ──────────────────────────────────
def engineer_features(df):
    """
    Create derived features that help the Bi-LSTM understand:
    - Catchment accumulation (rolling rainfall sums)
    - River momentum    (lag features)
    - Seasonal context  (cyclical month encoding)
    """
    # Rolling rainfall totals
    df["rain_3d_sum"]   = df["rainfall_mm"].rolling(3,  min_periods=1).sum()
    df["rain_7d_sum"]   = df["rainfall_mm"].rolling(7,  min_periods=1).sum()

    # Autoregressive lags
    df["level_lag1"]    = df["river_level_m"].shift(1)
    df["level_lag3"]    = df["river_level_m"].shift(3)

    # Rolling statistics on river level
    df["level_7d_mean"] = df["river_level_m"].rolling(7, min_periods=1).mean()
    df["level_7d_std"]  = (
        df["river_level_m"].rolling(7, min_periods=1).std().fillna(0.0)
    )

    # Cyclical month encoding
    # Why sin/cos? The model must know Dec→Jan is ONE step, not 11 steps back.
    month = df["date"].dt.month
    df["month_sin"] = np.sin(2.0 * np.pi * month / 12.0)
    df["month_cos"] = np.cos(2.0 * np.pi * month / 12.0)

    # Drop the few rows that have NaN from lag features
    df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)
    print(f"  After feature engineering: {len(df):,} rows, {len(FEATURE_COLS)} features")
    return df


# ── Step 3: Normalisation ─────────────────────────────────────────
def normalise(df, basin):
    """
    Min-Max scaling: X_scaled = (X − X_min) / (X_max − X_min)

    This maps every feature to [0, 1].
    The scaler is saved so the dashboard can inverse-transform
    predictions back into real metres.
    """
    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    df[FEATURE_COLS] = scaler.fit_transform(df[FEATURE_COLS])

    os.makedirs("models", exist_ok=True)
    scaler_path = f"models/scaler_{basin}.pkl"
    with open(scaler_path, "wb") as fh:
        pickle.dump(scaler, fh)
    print(f"  Scaler saved → {scaler_path}")
    return df, scaler


# ── Step 4 & 5: Sequences + split ────────────────────────────────
def create_sequences(df, basin):
    """
    Build sliding windows of length SEQUENCE_LEN (30 days).

    For each window of 30 consecutive days:
      X  =  feature matrix  shape (30, 12)
      y  =  next day's normalised river level  shape (,)

    Final array shapes:
      X_train: (n_train, 30, 12)
      y_train: (n_train,)
      … (same for val and test)
    """
    features = df[FEATURE_COLS].values   # shape: (n_days, 12)

    X_list, y_list = [], []
    for i in range(SEQUENCE_LEN, len(features)):
        X_list.append(features[i - SEQUENCE_LEN : i])
        y_list.append(features[i, TARGET_INDEX])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    n        = len(X)
    n_train  = int(n * TRAIN_FRAC)
    n_val    = int(n * VAL_FRAC)

    splits = {
        "X_train" : X[:n_train],
        "y_train" : y[:n_train],
        "X_val"   : X[n_train : n_train + n_val],
        "y_val"   : y[n_train : n_train + n_val],
        "X_test"  : X[n_train + n_val :],
        "y_test"  : y[n_train + n_val :],
    }

    print(f"  Train : {splits['X_train'].shape}")
    print(f"  Val   : {splits['X_val'].shape}")
    print(f"  Test  : {splits['X_test'].shape}")

    out = f"data/sequences_{basin}.npz"
    np.savez_compressed(out, **splits)
    print(f"  Sequences saved → {out}")


# ── Main ─────────────────────────────────────────────────────────
def run(basin):
    bar = "=" * 52
    print(f"\n{bar}\n  Processing: {basin.upper()}\n{bar}")
    df = load_and_clean(basin)
    df = engineer_features(df)
    df, scaler = normalise(df, basin)
    create_sequences(df, basin)


if __name__ == "__main__":
    for b in ["nzoia", "tana"]:
        run(b)
    print("\n✅  Done.  Next step:  python train_model.py")
