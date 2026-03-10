import pandas as pd
import numpy as np
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

GRAVITY_MS2 = 9.8
ACCEL_OUTLIER_ZSCORE = 4.0
AUDIO_MIN_DB = 30.0
AUDIO_MAX_DB = 120.0
SMOOTH_WINDOW = 3

AUDIO_SEVERITY = {
    "quiet":        0,
    "normal":       1,
    "conversation": 2,
    "loud":         3,
    "very_loud":    4,
    "argument":     5,
    "unknown":      1,
}

def compute_accel_magnitude(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    az_dynamic = df["accel_z"] - GRAVITY_MS2
    df["accel_magnitude"] = np.sqrt(
        df["accel_x"] ** 2 +
        df["accel_y"] ** 2 +
        az_dynamic ** 2
    )
    df["accel_lateral"] = np.sqrt(df["accel_x"] ** 2 + df["accel_y"] ** 2)
    return df

def clip_accel_outliers(df: pd.DataFrame) -> pd.DataFrame:
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
    df = df.copy().sort_values(["trip_id", "timestamp"]).reset_index(drop=True)
    for col in ["accel_magnitude", "accel_lateral"]:
        df[f"{col}_smooth"] = (
            df.groupby("trip_id")[col]
            .transform(lambda x: x.rolling(window=window, min_periods=1, center=True).mean())
        )
    logger.info(f"[accel] Smoothing applied (window={window}).")
    return df

def compute_accel_delta(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values(["trip_id", "timestamp"]).reset_index(drop=True)
    df["accel_delta"] = (
        df.groupby("trip_id")["accel_magnitude_smooth"]
        .transform(lambda x: x.diff().fillna(0))
    )
    return df

def preprocess_accelerometer(df: pd.DataFrame) -> pd.DataFrame:
    df = compute_accel_magnitude(df)
    df = clip_accel_outliers(df)
    df = smooth_accel(df)
    df = compute_accel_delta(df)
    logger.info(f"[accel] Preprocessing complete. Shape: {df.shape}")
    return df

def clean_audio_db(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["audio_level_db"] = df["audio_level_db"].clip(AUDIO_MIN_DB, AUDIO_MAX_DB)

    df["audio_level_db"] = df.groupby("trip_id")["audio_level_db"].transform(
        lambda x: x.fillna(x.median())
    )
    return df

def encode_audio_classification(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["audio_classification"] = df["audio_classification"].str.lower().str.strip()
    df["audio_severity_score"] = df["audio_classification"].map(AUDIO_SEVERITY).fillna(1)
    return df

def normalize_audio(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rng = AUDIO_MAX_DB - AUDIO_MIN_DB
    df["audio_db_norm"] = (df["audio_level_db"] - AUDIO_MIN_DB) / rng
    df["audio_db_log"] = np.log10(df["audio_level_db"].clip(lower=1))
    return df

def compute_audio_rolling_stats(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
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
    df = clean_audio_db(df)
    df = encode_audio_classification(df)
    df = normalize_audio(df)
    df = compute_audio_rolling_stats(df)
    logger.info(f"[audio] Preprocessing complete. Shape: {df.shape}")
    return df

def compute_trip_accel_baseline(df: pd.DataFrame) -> pd.DataFrame:
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
