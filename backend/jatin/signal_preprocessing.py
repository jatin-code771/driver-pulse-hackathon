"""
signal_preprocessing.py
------------------------
Cleans and preprocesses raw sensor signals (accelerometer + audio).

Key steps:
  1. Compute derived motion magnitude from 3-axis accelerometer
  2. Remove gravity component (z-axis baseline ≈ 9.8 m/s²)
  3. Detect and clip outliers using IQR / z-score
  4. Apply rolling-window smoothing to reduce noise
  5. Normalize audio dB levels
  6. Encode categorical audio classifications
  7. Compute per-trip baseline statistics for relative thresholding
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────
GRAVITY_MS2 = 9.8          # approximate gravity on z-axis (m/s²)
ACCEL_OUTLIER_ZSCORE = 4.0 # readings beyond this z-score are clipped
AUDIO_MIN_DB = 30.0        # physical minimum we trust
AUDIO_MAX_DB = 120.0       # physical maximum (pain threshold)
SMOOTH_WINDOW = 3          # rolling-average window (readings)

# Audio classification → ordinal severity score (0–6)
AUDIO_SEVERITY = {
    "quiet":        0,
    "normal":       1,
    "conversation": 2,
    "loud":         3,
    "very_loud":    4,
    "argument":     5,
    "unknown":      1,
}


# ──────────────────────────────────────────────────────────
# Accelerometer preprocessing
# ──────────────────────────────────────────────────────────

def compute_accel_magnitude(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute horizontal motion magnitude, removing the static gravity component
    from the z-axis.

    accel_magnitude = sqrt(accel_x² + accel_y² + (accel_z - g)²)

    A vehicle at rest on a level surface reads ~[0, 0, 9.8]; the net
    dynamic acceleration approaches zero.
    """
    df = df.copy()
    az_dynamic = df["accel_z"] - GRAVITY_MS2
    df["accel_magnitude"] = np.sqrt(
        df["accel_x"] ** 2 +
        df["accel_y"] ** 2 +
        az_dynamic ** 2
    )
    # Lateral-only magnitude (useful for detecting swerving/cornering)
    df["accel_lateral"] = np.sqrt(df["accel_x"] ** 2 + df["accel_y"] ** 2)
    return df


def clip_accel_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clip extreme acceleration outliers using a per-trip z-score approach.
    """
    df = df.copy()
    for col in ["accel_magnitude", "accel_x", "accel_y"]:
        grp_mean = df.groupby("trip_id")[col].transform("mean")
        grp_std  = df.groupby("trip_id")[col].transform("std").fillna(0)
        lower = grp_mean - ACCEL_OUTLIER_ZSCORE * grp_std
        upper = grp_mean + ACCEL_OUTLIER_ZSCORE * grp_std
        df[col] = df[col].clip(lower, upper)
    logger.info("[accel] Outlier clipping complete.")
    return df


def smooth_accel(df: pd.DataFrame, window: int = SMOOTH_WINDOW) -> pd.DataFrame:
    """
    Apply a rolling mean per trip to reduce measurement noise.
    """
    df = df.copy().sort_values(["trip_id", "timestamp"]).reset_index(drop=True)
    for col in ["accel_magnitude", "accel_lateral"]:
        df[f"{col}_smooth"] = (
            df.groupby("trip_id")[col]
            .transform(lambda x: x.rolling(window=window, min_periods=1, center=True).mean())
        )
    logger.info(f"[accel] Smoothing applied (window={window}).")
    return df


def compute_accel_delta(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute reading-to-reading change in magnitude (jerk proxy).
    """
    df = df.copy().sort_values(["trip_id", "timestamp"]).reset_index(drop=True)
    df["accel_delta"] = (
        df.groupby("trip_id")["accel_magnitude_smooth"]
        .transform(lambda x: x.diff().fillna(0))
    )
    return df


def preprocess_accelerometer(df: pd.DataFrame) -> pd.DataFrame:
    """Full accelerometer preprocessing pipeline."""
    df = compute_accel_magnitude(df)
    df = clip_accel_outliers(df)
    df = smooth_accel(df)
    df = compute_accel_delta(df)
    logger.info(f"[accel] Preprocessing complete. Shape: {df.shape}")
    return df


# ──────────────────────────────────────────────────────────
# Audio preprocessing
# ──────────────────────────────────────────────────────────

def clean_audio_db(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clip audio levels to the physically valid range [30, 120] dB.
    Replace NaN with median per trip.
    """
    df = df.copy()
    df["audio_level_db"] = df["audio_level_db"].clip(AUDIO_MIN_DB, AUDIO_MAX_DB)

    # Fill NaN with per-trip median
    df["audio_level_db"] = df.groupby("trip_id")["audio_level_db"].transform(
        lambda x: x.fillna(x.median())
    )
    return df


def encode_audio_classification(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map categorical audio classification to ordinal severity (0–5).
    Unknown labels default to 'normal' (1).
    """
    df = df.copy()
    df["audio_classification"] = df["audio_classification"].str.lower().str.strip()
    df["audio_severity_score"] = df["audio_classification"].map(AUDIO_SEVERITY).fillna(1)
    return df


def normalize_audio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize dB level to [0, 1] within the valid physical range.
    Also add a log-scaled version (more perceptually natural).
    """
    df = df.copy()
    rng = AUDIO_MAX_DB - AUDIO_MIN_DB
    df["audio_db_norm"] = (df["audio_level_db"] - AUDIO_MIN_DB) / rng
    # Human hearing is roughly logarithmic; log10 scale useful for thresholding
    df["audio_db_log"] = np.log10(df["audio_level_db"].clip(lower=1))
    return df


def compute_audio_rolling_stats(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """Per-trip rolling statistics on audio level."""
    df = df.copy().sort_values(["trip_id", "timestamp"]).reset_index(drop=True)
    df["audio_rolling_mean"] = (
        df.groupby("trip_id")["audio_level_db"]
        .transform(lambda x: x.rolling(window=window, min_periods=1).mean())
    )
    df["audio_rolling_max"] = (
        df.groupby("trip_id")["audio_level_db"]
        .transform(lambda x: x.rolling(window=window, min_periods=1).max())
    )
    df["audio_rolling_std"] = (
        df.groupby("trip_id")["audio_level_db"]
        .transform(lambda x: x.rolling(window=window, min_periods=1).std().fillna(0))
    )
    return df


def preprocess_audio(df: pd.DataFrame) -> pd.DataFrame:
    """Full audio preprocessing pipeline."""
    df = clean_audio_db(df)
    df = encode_audio_classification(df)
    df = normalize_audio(df)
    df = compute_audio_rolling_stats(df)
    logger.info(f"[audio] Preprocessing complete. Shape: {df.shape}")
    return df


# ──────────────────────────────────────────────────────────
# Trip-level baseline computation
# ──────────────────────────────────────────────────────────

def compute_trip_accel_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-trip acceleration statistics used for adaptive thresholding.
    Returns a trip-level summary DataFrame.
    """
    baseline = df.groupby("trip_id").agg(
        accel_mean=("accel_magnitude_smooth", "mean"),
        accel_std=("accel_magnitude_smooth", "std"),
        accel_p75=("accel_magnitude_smooth", lambda x: x.quantile(0.75)),
        accel_p95=("accel_magnitude_smooth", lambda x: x.quantile(0.95)),
        accel_max=("accel_magnitude_smooth", "max"),
        speed_mean=("speed_kmh", "mean"),
        speed_max=("speed_kmh", "max"),
        n_readings=("accel_magnitude_smooth", "count"),
    ).reset_index()
    baseline["accel_std"] = baseline["accel_std"].fillna(0)
    return baseline


def compute_trip_audio_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-trip audio statistics."""
    baseline = df.groupby("trip_id").agg(
        audio_mean=("audio_level_db", "mean"),
        audio_std=("audio_level_db", "std"),
        audio_p75=("audio_level_db", lambda x: x.quantile(0.75)),
        audio_p95=("audio_level_db", lambda x: x.quantile(0.95)),
        audio_max=("audio_level_db", "max"),
        sustained_max=("sustained_duration_sec", "max"),
        n_audio_readings=("audio_level_db", "count"),
    ).reset_index()
    baseline["audio_std"] = baseline["audio_std"].fillna(0)
    return baseline


if __name__ == "__main__":
    import sys
    from data_ingestion import load_all

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "./driver_pulse_hackathon_data"
    datasets = load_all(data_dir)

    acc = preprocess_accelerometer(datasets["accelerometer"])
    aud = preprocess_audio(datasets["audio"])

    acc_base = compute_trip_accel_baseline(acc)
    aud_base = compute_trip_audio_baseline(aud)

    print("\n── Accelerometer sample (preprocessed) ──")
    print(acc[["trip_id", "timestamp", "accel_magnitude", "accel_magnitude_smooth",
               "accel_delta", "speed_kmh"]].head(10).to_string(index=False))

    print("\n── Audio sample (preprocessed) ──")
    print(aud[["trip_id", "timestamp", "audio_level_db", "audio_db_norm",
               "audio_severity_score", "audio_rolling_mean"]].head(10).to_string(index=False))

    print("\n── Accel baseline (per trip) ──")
    print(acc_base.head(8).to_string(index=False))
