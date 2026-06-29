"""
generate_data.py
=================
Hydro-AI Sentinel — Kandy Otieno Genga | SCT213-C002-0106/2022
JKUAT BSc. Data Science 2026

PURPOSE:
  Generates 15 years (2009–2023) of realistic daily data mimicking:
  - CHIRPS satellite rainfall records
  - Water Resources Authority (WRA) river discharge records
  for both River Nzoia (Budalangi) and Tana River Basin.

RUN:
  python generate_data.py

OUTPUT:
  data/nzoia_dataset.csv
  data/tana_dataset.csv
"""

import numpy as np
import pandas as pd
import os

# ── Reproducibility ──────────────────────────────────────────────
np.random.seed(42)

START_DATE = "2009-01-01"
END_DATE   = "2023-12-31"


def seasonal_rainfall(doy, basin):
    """
    Simulate Kenya's two rainy seasons:
      Long Rains  : March – May  (peak ≈ April 15, DOY 105)
      Short Rains : Oct   – Dec  (peak ≈ Nov   1,  DOY 305)

    doy   : numpy array of day-of-year values
    basin : "nzoia" or "tana"
    """
    long_rains  = 18.0 * np.exp(-((doy - 105.0) ** 2) / (2.0 * 35.0 ** 2))
    short_rains = 12.0 * np.exp(-((doy - 305.0) ** 2) / (2.0 * 28.0 ** 2))
    base_noise  = np.random.exponential(1.5, size=len(doy))
    rain        = long_rains + short_rains + base_noise

    if basin == "tana":
        rain = rain * 1.15          # Tana highland catchment is wetter

    year_factor = np.random.uniform(0.6, 1.6)  # drought vs flood years
    return np.clip(rain * year_factor, 0.0, 120.0)


def build_river_level(rainfall, basin):
    """
    River level lags rainfall by 3–4 days and accumulates over a
    7-day rolling window (catchment retention effect).
    """
    lag = 3 if basin == "nzoia" else 4

    rolling_rain = (
        pd.Series(rainfall)
        .rolling(window=7, min_periods=1)
        .sum()
        .values
    )

    level = 1.5 + rolling_rain * 0.045
    level = np.roll(level, lag)
    level[:lag] = 1.5                          # clean head artefact

    noise = np.random.normal(0.0, 0.1, size=len(level))
    level = level + noise

    # Random flash-flood spikes (15 events over 15 years)
    for _ in range(15):
        idx = np.random.randint(0, len(level) - 5)
        level[idx : idx + 5] += np.random.uniform(1.5, 3.5)

    return np.clip(level, 0.5, 8.5)


def generate_dataset(basin):
    """Build and return the full 15-year daily DataFrame for one basin."""
    dates = pd.date_range(start=START_DATE, end=END_DATE, freq="D")
    n     = len(dates)
    doy   = dates.day_of_year.values.astype(float)

    # Generate rainfall year-by-year (each year gets its own variability factor)
    parts = []
    for yr in sorted(dates.year.unique()):
        mask = dates.year == yr
        parts.append(seasonal_rainfall(doy[mask], basin))
    rainfall_all = np.concatenate(parts)

    river_level = build_river_level(rainfall_all, basin)
    temperature = (22.0 + 6.0 * np.sin(2.0 * np.pi * doy / 365.0)
                   + np.random.normal(0.0, 1.0, n))
    humidity    = (65.0 + 20.0 * np.sin(2.0 * np.pi * (doy - 90.0) / 365.0)
                   + np.random.normal(0.0, 3.0, n))

    df = pd.DataFrame({
        "date"          : dates,
        "rainfall_mm"   : np.round(rainfall_all, 2),
        "river_level_m" : np.round(river_level,  3),
        "temperature_c" : np.round(np.clip(temperature, 15.0, 38.0), 1),
        "humidity_pct"  : np.round(np.clip(humidity,    30.0, 99.0), 1),
        "basin"         : basin,
    })

    # Inject ≈2 % missing values — mirrors real WRA data gaps
    miss_idx = np.random.choice(n, size=int(0.02 * n), replace=False)
    df.loc[miss_idx, "river_level_m"] = np.nan

    return df


def main():
    os.makedirs("data", exist_ok=True)

    for basin in ["nzoia", "tana"]:
        print(f"\n[{basin.upper()}] Generating dataset …")
        df = generate_dataset(basin)

        out = f"data/{basin}_dataset.csv"
        df.to_csv(out, index=False)

        missing = df["river_level_m"].isna().sum()
        print(f"  Rows      : {len(df):,}")
        print(f"  Missing   : {missing} river-level values (will be interpolated)")
        print(f"  Saved  →  {out}")

    print("\n✅  Done.  Next step:  python preprocess.py")


if __name__ == "__main__":
    main()
