"""
feature_engineering.py
Generates all model-ready features for the Driver Pulse system.
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# =========================================================
# MOTION FEATURES
# =========================================================

HARSH_BRAKE_THRESHOLD = 2.5
HARSH_ACCEL_THRESHOLD = 2.5
LATERAL_SWERVE_THRESHOLD = 1.5
MODERATE_EVENT_THRESHOLD = 1.5
MOTION_SCORE_CAP = 3.5


def engineer_motion_features(acc_df: pd.DataFrame) -> pd.DataFrame:

    df = acc_df.copy()

    df["is_harsh_brake"] = (
        (df["accel_magnitude_smooth"] >= HARSH_BRAKE_THRESHOLD)
        & (df["accel_delta"] <= 0)
    ).astype(int)

    df["is_harsh_accel"] = (
        (df["accel_magnitude_smooth"] >= HARSH_ACCEL_THRESHOLD)
        & (df["accel_delta"] > 0)
    ).astype(int)

    df["is_lateral_swerve"] = (df["accel_lateral"] > LATERAL_SWERVE_THRESHOLD).astype(
        int
    )

    df["is_moderate_event"] = (
        (df["accel_magnitude_smooth"] > MODERATE_EVENT_THRESHOLD)
        & (df["is_harsh_brake"] == 0)
        & (df["is_harsh_accel"] == 0)
    ).astype(int)

    df["motion_score"] = (df["accel_magnitude_smooth"] / MOTION_SCORE_CAP).clip(0, 1)

    max_delta = df["accel_delta"].abs().max() or 1.0
    df["jerk_score"] = (df["accel_delta"].abs() / max_delta).clip(0, 1)

    df["is_low_speed"] = (df["speed_kmh"] < 20).astype(int)
    df["is_stationary"] = (df["speed_kmh"] == 0).astype(int)

    df = df.sort_values(["trip_id", "timestamp"]).reset_index(drop=True)

    df["harsh_event_rolling5"] = (
        df.groupby("trip_id")
        .apply(
            lambda g: (
                g["is_harsh_brake"] | g["is_harsh_accel"]
            ).rolling(5, min_periods=1).sum()
        )
        .reset_index(level=0, drop=True)
    ).fillna(0)

    return df


# =========================================================
# AUDIO FEATURES
# =========================================================

AUDIO_SPIKE_THRESHOLD_DB = 85.0
AUDIO_ARGUMENT_THRESHOLD = 4
SUSTAINED_NOISE_THRESHOLD = 60


def engineer_audio_features(aud_df: pd.DataFrame) -> pd.DataFrame:

    df = aud_df.copy()

    df["is_noise_spike"] = (df["audio_level_db"] >= AUDIO_SPIKE_THRESHOLD_DB).astype(
        int
    )

    df["is_argument_signal"] = (
        df["audio_severity_score"] >= AUDIO_ARGUMENT_THRESHOLD
    ).astype(int)

    df["is_sustained_noise"] = (
        df["sustained_duration_sec"] >= SUSTAINED_NOISE_THRESHOLD
    ).astype(int)

    db_score = ((df["audio_level_db"] - 30) / 75).clip(0, 1)
    sev_score = (df["audio_severity_score"] / 5).clip(0, 1)

    df["audio_score"] = (0.6 * db_score + 0.4 * sev_score).clip(0, 1)

    df["sustained_penalty"] = df["sustained_duration_sec"].clip(0, 300) / 300

    df["audio_score_adjusted"] = (
        df["audio_score"] * (1 + 0.4 * df["sustained_penalty"])
    ).clip(0, 1)

    return df


# =========================================================
# SIGNAL FUSION
# =========================================================

MOTION_WEIGHT = 0.55
AUDIO_WEIGHT = 0.45

HIGH_STRESS_THRESHOLD = 0.70
MEDIUM_STRESS_THRESHOLD = 0.45


def fuse_signals(acc_df, aud_df):

    acc = acc_df.copy()
    aud = aud_df.copy()

    acc["timestamp"] = pd.to_datetime(acc["timestamp"], errors="coerce")
    aud["timestamp"] = pd.to_datetime(aud["timestamp"], errors="coerce")

    fused = acc.merge(aud, on=["trip_id", "timestamp"], how="left")

    # Ensure audio columns always exist
    for col in [
        "is_argument_signal",
        "is_noise_spike",
        "is_sustained_noise",
        "audio_score_adjusted",
    ]:
        if col not in fused.columns:
            fused[col] = 0

    fused["audio_score_adjusted"] = fused["audio_score_adjusted"].fillna(0)

    fused["combined_score"] = (
        MOTION_WEIGHT * fused["motion_score"]
        + AUDIO_WEIGHT * fused["audio_score_adjusted"]
    ).clip(0, 1)

    fused["stress_severity"] = pd.cut(
        fused["combined_score"],
        bins=[-np.inf, MEDIUM_STRESS_THRESHOLD, HIGH_STRESS_THRESHOLD, np.inf],
        labels=["low", "medium", "high"],
    )

    return fused


# =========================================================
# TEMPORAL FEATURES
# =========================================================

def engineer_temporal_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    ts = df["timestamp"]

    df["hour_of_day"] = ts.dt.hour
    df["minute_of_day"] = ts.dt.hour * 60 + ts.dt.minute

    df["is_rush_hour"] = ts.dt.hour.isin([7, 8, 9, 17, 18, 19]).astype(int)
    df["is_night"] = ts.dt.hour.isin(list(range(22, 24)) + list(range(0, 6))).astype(
        int
    )

    df = df.sort_values(["trip_id", "timestamp"]).reset_index(drop=True)

    max_elapsed = df.groupby("trip_id")["elapsed_seconds"].transform("max").replace(
        0, np.nan
    )

    df["trip_progress_frac"] = (df["elapsed_seconds"] / max_elapsed).fillna(0)

    df["near_trip_start"] = (df["trip_progress_frac"] < 0.15).astype(int)
    df["near_trip_end"] = (df["trip_progress_frac"] > 0.85).astype(int)

    return df


# =========================================================
# TRIP FEATURE MATRIX
# =========================================================

def build_trip_feature_matrix(
    fused_df: pd.DataFrame,
    aud_df: pd.DataFrame,
    trips_df: Optional[pd.DataFrame] = None,
):

    motion_agg = fused_df.groupby("trip_id").agg(
        n_accel_readings=("accel_magnitude_smooth", "count"),
        mean_accel=("accel_magnitude_smooth", "mean"),
        max_accel=("accel_magnitude_smooth", "max"),
        n_harsh_brakes=("is_harsh_brake", "sum"),
        n_harsh_accels=("is_harsh_accel", "sum"),
        n_swerves=("is_lateral_swerve", "sum"),
        motion_score_mean=("motion_score", "mean"),
        motion_score_max=("motion_score", "max"),
        mean_speed_kmh=("speed_kmh", "mean"),
        max_speed_kmh=("speed_kmh", "max"),
        n_low_speed_events=("is_low_speed", "sum"),
    ).reset_index()

    audio_agg = aud_df.groupby("trip_id").agg(
        n_audio_readings=("audio_level_db", "count"),
        mean_audio_db=("audio_level_db", "mean"),
        max_audio_db=("audio_level_db", "max"),
        n_noise_spikes=("is_noise_spike", "sum"),
        n_argument_signals=("is_argument_signal", "sum"),
        n_sustained_noise=("is_sustained_noise", "sum"),
        audio_score_mean=("audio_score_adjusted", "mean"),
        audio_score_max=("audio_score_adjusted", "max"),
        total_sustained_sec=("sustained_duration_sec", "sum"),
    ).reset_index()

    stress_agg = fused_df.groupby("trip_id").agg(
        combined_score_mean=("combined_score", "mean"),
        combined_score_max=("combined_score", "max"),
        n_high_stress=("stress_severity", lambda x: (x == "high").sum()),
        n_medium_stress=("stress_severity", lambda x: (x == "medium").sum()),
    ).reset_index()

    trip_features = (
        motion_agg.merge(audio_agg, on="trip_id", how="outer")
        .merge(stress_agg, on="trip_id", how="outer")
    )

    if trips_df is not None:
        trip_features = trip_features.merge(trips_df, on="trip_id", how="left")

    trip_features["harsh_event_rate"] = (
        (trip_features["n_harsh_brakes"] + trip_features["n_harsh_accels"])
        / trip_features["n_accel_readings"].replace(0, np.nan)
    ).fillna(0)

    trip_features["conflict_signal_rate"] = (
        (trip_features["n_noise_spikes"] + trip_features["n_argument_signals"])
        / trip_features["n_audio_readings"].replace(0, np.nan)
    ).fillna(0)

    trip_features["trip_quality_score"] = (
        1
        - 0.35 * trip_features["harsh_event_rate"]
        - 0.35 * trip_features["conflict_signal_rate"]
        - 0.30 * trip_features["combined_score_mean"]
    ).clip(0, 1)

    return trip_features.fillna(0)


# =========================================================
# PIPELINE RUNNER
# =========================================================

def build_all_features(datasets: dict):

    from signal_preprocessing import preprocess_accelerometer, preprocess_audio

    acc = preprocess_accelerometer(datasets["accelerometer"])
    aud = preprocess_audio(datasets["audio"])

    acc = engineer_motion_features(acc)
    acc = engineer_temporal_features(acc)

    aud = engineer_audio_features(aud)

    fused = fuse_signals(acc, aud)

    trip_feat = build_trip_feature_matrix(
        fused,
        aud,
        datasets.get("trips"),
    )

    return {
        "acc_features": acc,
        "aud_features": aud,
        "fused_features": fused,
        "trip_features": trip_feat,
    }