"""
feature_engineering.py
-----------------------
Generates all model-ready features for the Driver Pulse system.

Feature groups produced:
  A. Motion features  – harsh braking / aggressive accel / swerve detection
  B. Audio features   – noise spikes, sustained high audio, argument signals
  C. Fusion features  – combined motion+audio stress scores
  D. Temporal features– time-of-day, trip progress fraction, elapsed windows
  E. Earnings features– velocity, goal gap, pacing score, trajectory label
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────
# A.  Motion Feature Engineering
# ──────────────────────────────────────────────────────────

# Fixed thresholds (m/s² net dynamic acceleration after gravity removal)
HARSH_BRAKE_THRESHOLD      = 2.0   # sudden deceleration / forward lurch
HARSH_ACCEL_THRESHOLD      = 2.0   # hard acceleration
LATERAL_SWERVE_THRESHOLD   = 1.5   # sharp cornering / lane change
MODERATE_EVENT_THRESHOLD   = 1.2   # softer events worth tracking


def engineer_motion_features(acc_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds binary and graded motion event features.

    Assumes acc_df has been through signal_preprocessing.preprocess_accelerometer().
    """
    df = acc_df.copy()

    # ── Binary event flags ────────────────────────────────
    df["is_harsh_brake"]  = (
        (df["accel_delta"] < -HARSH_BRAKE_THRESHOLD)
    ).astype(int)

    df["is_harsh_accel"]  = (
        (df["accel_delta"] > HARSH_ACCEL_THRESHOLD)
    ).astype(int)

    df["is_lateral_swerve"] = (
        df["accel_lateral"] > LATERAL_SWERVE_THRESHOLD
    ).astype(int)

    df["is_moderate_event"] = (
        (df["accel_magnitude_smooth"] > MODERATE_EVENT_THRESHOLD) &
        (df["is_harsh_brake"] == 0) &
        (df["is_harsh_accel"] == 0)
    ).astype(int)

    # ── Graded motion score [0, 1] ────────────────────────
    # Normalise smoothed magnitude to a 0-1 score.
    # Cap at 10 m/s² (well beyond any real driving event).
    df["motion_score"] = (df["accel_magnitude_smooth"] / 10.0).clip(0, 1)

    # ── Jerk score: penalise rapid *changes* in acceleration ──
    # More human-perceptible discomfort comes from jerk, not magnitude alone.
    max_delta = df["accel_delta"].abs().max() or 1.0
    df["jerk_score"] = (df["accel_delta"].abs() / max_delta).clip(0, 1)

    # ── Speed context ─────────────────────────────────────
    # Low-speed harsh events are more likely passenger incidents than traffic.
    df["is_low_speed"] = (df["speed_kmh"] < 20).astype(int)
    df["is_stationary"] = (df["speed_kmh"] == 0).astype(int)

    df = df.sort_values(["trip_id", "timestamp"]).reset_index(drop=True)
    df["harsh_event_rolling5"] = (
        df.groupby("trip_id")
        .apply(lambda g: (g["is_harsh_brake"] | g["is_harsh_accel"]).rolling(5, min_periods=1).sum())
        .reset_index(level=0, drop=True)
    ).fillna(0)

    logger.info(
        f"[motion features] harsh_brake={df['is_harsh_brake'].sum()} | "
        f"harsh_accel={df['is_harsh_accel'].sum()} | "
        f"swerve={df['is_lateral_swerve'].sum()}"
    )
    return df


def classify_motion_event(row: pd.Series) -> str:
    """Return a human-readable motion event label for a single row."""
    if row["is_harsh_brake"]:
        return "harsh_brake"
    if row["is_harsh_accel"]:
        return "harsh_accel"
    if row["is_lateral_swerve"]:
        return "swerve"
    if row["is_moderate_event"]:
        return "moderate_event"
    return "normal"


# ──────────────────────────────────────────────────────────
# B.  Audio Feature Engineering
# ──────────────────────────────────────────────────────────

AUDIO_SPIKE_THRESHOLD_DB   = 85.0   # dB — loud enough to indicate conflict
AUDIO_ARGUMENT_THRESHOLD   = 4      # severity score ≥ 4 = very_loud or argument
SUSTAINED_NOISE_THRESHOLD  = 60     # seconds of continuous high audio


def engineer_audio_features(aud_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds binary and graded audio event features.

    Assumes aud_df has been through signal_preprocessing.preprocess_audio().
    """
    df = aud_df.copy()

    # ── Binary flags ──────────────────────────────────────
    df["is_noise_spike"] = (
        df["audio_level_db"] >= AUDIO_SPIKE_THRESHOLD_DB
    ).astype(int)

    df["is_argument_signal"] = (
        df["audio_severity_score"] >= AUDIO_ARGUMENT_THRESHOLD
    ).astype(int)

    df["is_sustained_noise"] = (
        df["sustained_duration_sec"] >= SUSTAINED_NOISE_THRESHOLD
    ).astype(int)

    # ── Graded audio score [0, 1] ─────────────────────────
    # dB is logarithmic; map 30–120 dB linearly then blend with severity.
    db_score = ((df["audio_level_db"] - 30) / 90).clip(0, 1)
    sev_score = (df["audio_severity_score"] / 5).clip(0, 1)
    df["audio_score"] = (0.6 * db_score + 0.4 * sev_score).clip(0, 1)

    # ── Sustained noise penalty ───────────────────────────
    # Longer sustained noise should amplify the score.
    # Cap at 300 s (5 minutes) for normalisation.
    df["sustained_penalty"] = (
        df["sustained_duration_sec"].clip(0, 300) / 300
    )
    df["audio_score_adjusted"] = (
        df["audio_score"] * (1 + 0.3 * df["sustained_penalty"])
    ).clip(0, 1)

    # ── Trip-level context ────────────────────────────────
    df["audio_above_baseline"] = (
        df["audio_level_db"] > df["audio_rolling_mean"] + df["audio_rolling_std"]
    ).astype(int)

    logger.info(
        f"[audio features] noise_spike={df['is_noise_spike'].sum()} | "
        f"argument_signal={df['is_argument_signal'].sum()} | "
        f"sustained={df['is_sustained_noise'].sum()}"
    )
    return df


def classify_audio_event(row: pd.Series) -> str:
    """Return a human-readable audio event label for a single row."""
    if row["is_argument_signal"] and row["is_sustained_noise"]:
        return "sustained_argument"
    if row["is_argument_signal"]:
        return "argument_signal"
    if row["is_sustained_noise"]:
        return "sustained_noise"
    if row["is_noise_spike"]:
        return "noise_spike"
    return "normal"


# ──────────────────────────────────────────────────────────
# C.  Fusion / Combined Stress Score
# ──────────────────────────────────────────────────────────

MOTION_WEIGHT = 0.55   # motion slightly more diagnostic for driving safety
AUDIO_WEIGHT  = 0.45

HIGH_STRESS_THRESHOLD  = 0.70
MEDIUM_STRESS_THRESHOLD = 0.45


def fuse_signals(
    acc_df: pd.DataFrame,
    aud_df: pd.DataFrame,
    tolerance_sec: int = 60,
) -> pd.DataFrame:
    """
    Time-align accelerometer and audio readings and compute a combined
    stress score for every accelerometer timestamp.

    Strategy:
      - For each accelerometer reading, find the nearest audio reading
        within ±tolerance_sec on the same trip.
      - Fuse the two scores: combined = w_m * motion + w_a * audio
      - If no audio match found, use motion score alone (downweighted).

    Returns an enriched accelerometer DataFrame with audio columns attached.
    """
    acc = acc_df.copy()
    aud = aud_df[["trip_id", "timestamp", "audio_score_adjusted",
                   "is_noise_spike", "is_argument_signal",
                   "is_sustained_noise", "audio_level_db",
                   "audio_severity_score"]].copy()
    aud = aud.rename(columns={"timestamp": "audio_ts"})

    merged_rows = []

    for trip_id, acc_grp in acc.groupby("trip_id"):
        aud_grp = aud[aud["trip_id"] == trip_id].copy()

        if aud_grp.empty:
            # No audio for this trip – use motion score only (penalised)
            acc_grp = acc_grp.copy()
            acc_grp["audio_score_fused"]     = 0.0
            acc_grp["is_noise_spike"]        = 0
            acc_grp["is_argument_signal"]    = 0
            acc_grp["is_sustained_noise"]    = 0
            acc_grp["audio_level_db_fused"]  = np.nan
            merged_rows.append(acc_grp)
            continue

        aud_ts = aud_grp["audio_ts"].values.astype("datetime64[s]").astype(np.int64)

        for _, acc_row in acc_grp.iterrows():
            acc_ts_ns = np.int64(acc_row["timestamp"].value // 1_000_000_000)
            deltas = np.abs(aud_ts - acc_ts_ns)
            closest_idx = deltas.argmin()
            closest_delta = deltas[closest_idx]

            if closest_delta <= tolerance_sec:
                aud_row = aud_grp.iloc[closest_idx]
                acc_grp.loc[acc_row.name, "audio_score_fused"]    = aud_row["audio_score_adjusted"]
                acc_grp.loc[acc_row.name, "is_noise_spike"]       = aud_row["is_noise_spike"]
                acc_grp.loc[acc_row.name, "is_argument_signal"]   = aud_row["is_argument_signal"]
                acc_grp.loc[acc_row.name, "is_sustained_noise"]   = aud_row["is_sustained_noise"]
                acc_grp.loc[acc_row.name, "audio_level_db_fused"] = aud_row["audio_level_db"]
            else:
                acc_grp.loc[acc_row.name, "audio_score_fused"]    = 0.0
                acc_grp.loc[acc_row.name, "is_noise_spike"]       = 0
                acc_grp.loc[acc_row.name, "is_argument_signal"]   = 0
                acc_grp.loc[acc_row.name, "is_sustained_noise"]   = 0
                acc_grp.loc[acc_row.name, "audio_level_db_fused"] = np.nan

        merged_rows.append(acc_grp)

    fused = pd.concat(merged_rows, ignore_index=True)

    # ── Combined score ────────────────────────────────────
    has_audio = fused["audio_score_fused"] > 0
    fused["combined_score"] = np.where(
        has_audio,
        MOTION_WEIGHT * fused["motion_score"] + AUDIO_WEIGHT * fused["audio_score_fused"],
        fused["motion_score"] * 0.8,   # slight penalty for no audio confirmation
    )
    fused["combined_score"] = fused["combined_score"].clip(0, 1)

    # ── Stress severity label ─────────────────────────────
    fused["stress_severity"] = pd.cut(
        fused["combined_score"],
        bins=[-np.inf, MEDIUM_STRESS_THRESHOLD, HIGH_STRESS_THRESHOLD, np.inf],
        labels=["low", "medium", "high"],
    )

    logger.info(
        f"[fusion] Rows: {len(fused)} | "
        f"high stress: {(fused['stress_severity'] == 'high').sum()} | "
        f"medium: {(fused['stress_severity'] == 'medium').sum()}"
    )
    return fused


# ──────────────────────────────────────────────────────────
# D.  Temporal Feature Engineering
# ──────────────────────────────────────────────────────────

def engineer_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time-aware features that capture when during the day/trip an event occurs.
    """
    df = df.copy()
    ts = df["timestamp"]

    df["hour_of_day"]    = ts.dt.hour
    df["minute_of_day"]  = ts.dt.hour * 60 + ts.dt.minute
    df["is_rush_hour"]   = ts.dt.hour.isin([7, 8, 9, 17, 18, 19]).astype(int)
    df["is_night"]       = ts.dt.hour.isin(list(range(22, 24)) + list(range(0, 6))).astype(int)

    df = df.sort_values(["trip_id", "timestamp"]).reset_index(drop=True)
    max_elapsed = df.groupby("trip_id")["elapsed_seconds"].transform("max").replace(0, np.nan)
    df["trip_progress_frac"] = (df["elapsed_seconds"] / max_elapsed).fillna(0)

    # Event proximity to trip start/end (conflicts often cluster at start/end)
    df["near_trip_start"] = (df["trip_progress_frac"] < 0.15).astype(int)
    df["near_trip_end"]   = (df["trip_progress_frac"] > 0.85).astype(int)

    return df


# ──────────────────────────────────────────────────────────
# E.  Earnings Feature Engineering
# ──────────────────────────────────────────────────────────

MIN_HOURS_FOR_VELOCITY = 0.25   # 15-minute cold-start guard


def engineer_earnings_features(goals_df: pd.DataFrame) -> pd.DataFrame:
    """
    Enriches the driver_goals DataFrame with:
      - earnings_velocity_calc : re-derived from raw fields
      - earnings_gap           : how much more is needed to hit target
      - pacing_ratio           : actual velocity / required velocity to finish on time
      - shift_hours_remaining  : hours left in scheduled shift
      - goal_completion_pct    : how far along toward target
      - trajectory_label       : categorical forecast (ahead / on_track / at_risk / critical)
    """
    df = goals_df.copy()

    # ── Re-derive velocity robustly ───────────────────────
    # Use provided field but fall back to computation if missing / zero.
    if "earnings_velocity" in df.columns:
        mask_zero = df["earnings_velocity"].isna() | (df["earnings_velocity"] == 0)
        df.loc[mask_zero, "earnings_velocity"] = (
            df.loc[mask_zero, "current_earnings"] /
            df.loc[mask_zero, "current_hours"].clip(lower=MIN_HOURS_FOR_VELOCITY)
        )

    # ── Goal gap ──────────────────────────────────────────
    df["earnings_gap"] = (df["target_earnings"] - df["current_earnings"]).clip(lower=0)

    # ── Goal completion % ─────────────────────────────────
    df["goal_completion_pct"] = (
        df["current_earnings"] / df["target_earnings"].replace(0, np.nan)
    ).clip(0, 1.0)

    # ── Required velocity to finish on time ───────────────
    # Parse shift times if available
    for col in ["shift_start_time", "shift_end_time"]:
        if col not in df.columns:
            df[col] = np.nan

    # Approximate remaining shift hours from target_hours and current_hours
    df["hours_elapsed"] = df["current_hours"].clip(lower=0)
    df["hours_remaining"] = (
        df["target_hours"] - df["hours_elapsed"]
    ).clip(lower=0)

    df["required_velocity"] = np.where(
        df["hours_remaining"] > 0,
        df["earnings_gap"] / df["hours_remaining"],
        np.inf,
    )
    # Clip infinite (shift over but goal not met) to a large number for sorting
    df["required_velocity"] = df["required_velocity"].replace(np.inf, 9999)

    # ── Pacing ratio ──────────────────────────────────────
    # > 1.0 → ahead of pace; < 1.0 → falling behind
    df["pacing_ratio"] = np.where(
        df["required_velocity"] > 0,
        df["earnings_velocity"] / df["required_velocity"].replace(0, np.nan),
        np.nan,
    )
    df["pacing_ratio"] = df["pacing_ratio"].clip(0, 5)

    # ── Cold-start flag ───────────────────────────────────
    df["is_cold_start"] = (df["current_hours"] < MIN_HOURS_FOR_VELOCITY).astype(int)

    # ── Trajectory label ──────────────────────────────────
    def _label(row):
        if row["goal_completion_pct"] >= 1.0:
            return "achieved"
        if row["is_cold_start"]:
            return "cold_start"
        pr = row["pacing_ratio"]
        if pd.isna(pr):
            return "unknown"
        if pr >= 1.25:
            return "ahead"
        if pr >= 0.85:
            return "on_track"
        if pr >= 0.50:
            return "at_risk"
        return "critical"

    df["trajectory_label"] = df.apply(_label, axis=1)

    # ── Earnings forecast at end of shift ─────────────────
    df["projected_earnings"] = (
        df["current_earnings"] +
        df["earnings_velocity"] * df["hours_remaining"]
    )
    df["projected_completion_pct"] = (
        df["projected_earnings"] / df["target_earnings"].replace(0, np.nan)
    ).clip(0, 2.0)

    logger.info(
        f"[earnings features] "
        f"on_track={( df['trajectory_label'] == 'on_track').sum()} | "
        f"at_risk={(df['trajectory_label'] == 'at_risk').sum()} | "
        f"critical={(df['trajectory_label'] == 'critical').sum()}"
    )
    return df


# ──────────────────────────────────────────────────────────
# F.  Trip-level Summary Feature Matrix
# ──────────────────────────────────────────────────────────

def build_trip_feature_matrix(
    fused_df: pd.DataFrame,
    aud_df: pd.DataFrame,
    trips_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Aggregate all per-reading features up to the trip level.
    Returns one row per trip with rich summary statistics.
    """
    # ── Motion aggregates ─────────────────────────────────
    motion_agg = fused_df.groupby("trip_id").agg(
        n_accel_readings      = ("accel_magnitude_smooth", "count"),
        mean_accel            = ("accel_magnitude_smooth", "mean"),
        max_accel             = ("accel_magnitude_smooth", "max"),
        n_harsh_brakes        = ("is_harsh_brake", "sum"),
        n_harsh_accels        = ("is_harsh_accel", "sum"),
        n_swerves             = ("is_lateral_swerve", "sum"),
        motion_score_mean     = ("motion_score", "mean"),
        motion_score_max      = ("motion_score", "max"),
        mean_speed_kmh        = ("speed_kmh", "mean"),
        max_speed_kmh         = ("speed_kmh", "max"),
        n_low_speed_events    = ("is_low_speed", "sum"),
    ).reset_index()

    # ── Audio aggregates ──────────────────────────────────
    audio_agg = aud_df.groupby("trip_id").agg(
        n_audio_readings    = ("audio_level_db", "count"),
        mean_audio_db       = ("audio_level_db", "mean"),
        max_audio_db        = ("audio_level_db", "max"),
        n_noise_spikes      = ("is_noise_spike", "sum"),
        n_argument_signals  = ("is_argument_signal", "sum"),
        n_sustained_noise   = ("is_sustained_noise", "sum"),
        audio_score_mean    = ("audio_score_adjusted", "mean"),
        audio_score_max     = ("audio_score_adjusted", "max"),
        total_sustained_sec = ("sustained_duration_sec", "sum"),
    ).reset_index()

    # ── Combined stress aggregates ────────────────────────
    stress_agg = fused_df.groupby("trip_id").agg(
        combined_score_mean = ("combined_score", "mean"),
        combined_score_max  = ("combined_score", "max"),
        n_high_stress       = ("stress_severity", lambda x: (x == "high").sum()),
        n_medium_stress     = ("stress_severity", lambda x: (x == "medium").sum()),
        n_conflict_moments  = ("is_argument_signal", "sum"),
    ).reset_index()

    # ── Merge everything ──────────────────────────────────
    trip_features = (
        motion_agg
        .merge(audio_agg,  on="trip_id", how="outer")
        .merge(stress_agg, on="trip_id", how="outer")
    )

    if trips_df is not None and not trips_df.empty:
        trip_features = trip_features.merge(trips_df, on="trip_id", how="left")

    # ── Derived trip-level ratios ─────────────────────────
    trip_features["harsh_event_rate"] = (
        (trip_features["n_harsh_brakes"] + trip_features["n_harsh_accels"]) /
        trip_features["n_accel_readings"].replace(0, np.nan)
    ).fillna(0)

    trip_features["conflict_signal_rate"] = (
        (trip_features["n_noise_spikes"] + trip_features["n_argument_signals"]) /
        trip_features["n_audio_readings"].replace(0, np.nan)
    ).fillna(0)

    trip_features["trip_quality_score"] = (
        1.0
        - 0.35 * trip_features["harsh_event_rate"].clip(0, 1)
        - 0.35 * trip_features["conflict_signal_rate"].clip(0, 1)
        - 0.30 * trip_features["combined_score_mean"].clip(0, 1)
    ).clip(0, 1)

    logger.info(f"[trip features] {len(trip_features)} trips in feature matrix.")
    return trip_features.fillna(0)


# ──────────────────────────────────────────────────────────
# Convenience runner
# ──────────────────────────────────────────────────────────

def build_all_features(datasets: dict) -> dict:
    """
    Run the complete feature engineering pipeline.

    Parameters
    ----------
    datasets : dict returned by data_ingestion.load_all()

    Returns
    -------
    dict with keys:
      acc_features     – per-reading accelerometer features
      aud_features     – per-reading audio features
      fused_features   – aligned & fused per-reading features
      trip_features    – trip-level feature matrix
      earnings_features– driver goal & earnings features
    """
    from signal_preprocessing import preprocess_accelerometer, preprocess_audio

    acc = preprocess_accelerometer(datasets["accelerometer"])
    aud = preprocess_audio(datasets["audio"])

    acc = engineer_motion_features(acc)
    acc = engineer_temporal_features(acc)

    aud = engineer_audio_features(aud)

    fused = fuse_signals(acc, aud)

    trip_feat = build_trip_feature_matrix(fused, aud, datasets.get("trips"))

    earnings_feat = engineer_earnings_features(datasets["driver_goals"])

    return {
        "acc_features":      acc,
        "aud_features":      aud,
        "fused_features":    fused,
        "trip_features":     trip_feat,
        "earnings_features": earnings_feat,
    }


if __name__ == "__main__":
    import sys
    from data_ingestion import load_all

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "./driver_pulse_hackathon_data"
    datasets = load_all(data_dir)
    features = build_all_features(datasets)

    print("\n══ Fused signal sample ══")
    cols = ["trip_id", "timestamp", "motion_score", "audio_score_fused",
            "combined_score", "stress_severity", "is_harsh_brake", "is_argument_signal"]
    print(features["fused_features"][cols].head(15).to_string(index=False))

    print("\n══ Trip feature matrix (top rows) ══")
    print(features["trip_features"].head(8).to_string(index=False))

    print("\n══ Earnings features sample ══")
    ecols = ["driver_id", "earnings_velocity", "pacing_ratio",
             "earnings_gap", "projected_completion_pct", "trajectory_label"]
    print(features["earnings_features"][ecols].head(10).to_string(index=False))
