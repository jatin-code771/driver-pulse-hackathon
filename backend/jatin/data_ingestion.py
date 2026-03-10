import pandas as pd
import numpy as np
import os
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

SCHEMAS = {
    "accelerometer": {
        "required": ["sensor_id", "trip_id", "timestamp", "elapsed_seconds",
                     "accel_x", "accel_y", "accel_z", "speed_kmh", "gps_lat", "gps_lon"],
        "numeric":  ["elapsed_seconds", "accel_x", "accel_y", "accel_z",
                     "speed_kmh", "gps_lat", "gps_lon"],
    },
    "audio": {
        "required": ["audio_id", "trip_id", "timestamp", "elapsed_seconds",
                     "audio_level_db", "audio_classification", "sustained_duration_sec"],
        "numeric":  ["elapsed_seconds", "audio_level_db", "sustained_duration_sec"],
    },
    "trips": {
        "required": ["trip_id", "driver_id"],
        "numeric":  [],
    },
    "drivers": {
        "required": ["driver_id"],
        "numeric":  [],
    },
    "driver_goals": {
        "required": ["goal_id", "driver_id", "target_earnings", "current_earnings",
                     "current_hours", "target_hours", "earnings_velocity"],
        "numeric":  ["target_earnings", "current_earnings", "current_hours",
                     "target_hours", "earnings_velocity"],
    },
    "earnings_velocity": {
        "required": ["driver_id"],
        "numeric":  [],
    },
}

def _validate_schema(df: pd.DataFrame, name: str) -> pd.DataFrame:
    schema = SCHEMAS.get(name, {})
    missing_cols = [c for c in schema.get("required", []) if c not in df.columns]
    if missing_cols:
        logger.warning(f"[{name}] Missing expected columns: {missing_cols}")
    for col in schema.get("numeric", []):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def load_accelerometer(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = _validate_schema(df, "accelerometer")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    n_bad = df["timestamp"].isna().sum()
    if n_bad:
        logger.warning(f"[accelerometer] {n_bad} rows with unparseable timestamps dropped.")
        df = df.dropna(subset=["timestamp"])
    df = df.sort_values(["trip_id", "timestamp"]).reset_index(drop=True)
    logger.info(f"[accelerometer] Loaded {len(df)} rows across {df['trip_id'].nunique()} trips.")
    return df

def load_audio(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = _validate_schema(df, "audio")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df.sort_values(["trip_id", "timestamp"]).reset_index(drop=True)
    logger.info(f"[audio] Loaded {len(df)} rows across {df['trip_id'].nunique()} trips.")
    return df

def load_trips(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = _validate_schema(df, "trips")
    logger.info(f"[trips] Loaded {len(df)} trips.")
    return df

def load_drivers(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = _validate_schema(df, "drivers")
    logger.info(f"[drivers] Loaded {len(df)} driver profiles.")
    return df

def load_driver_goals(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = _validate_schema(df, "driver_goals")
    logger.info(f"[driver_goals] Loaded {len(df)} goal records.")
    return df

def load_earnings_velocity(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = _validate_schema(df, "earnings_velocity")
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    logger.info(f"[earnings_velocity] Loaded {len(df)} velocity checkpoints.")
    return df

def load_all(data_dir: str) -> dict:
    base = Path(data_dir)
    datasets = {}

    paths = {
        "accelerometer":    base / "sensor_data"    / "accelerometer_data.csv",
        "audio":            base / "sensor_data"    / "audio_intensity_data.csv",
        "trips":            base / "trips"           / "trips.csv",
        "drivers":          base / "drivers"         / "drivers.csv",
        "driver_goals":     base / "earnings"        / "driver_goals.csv",
        "earnings_velocity":base / "earnings"        / "earnings_velocity_log.csv",
    }

    loaders = {
        "accelerometer":    load_accelerometer,
        "audio":            load_audio,
        "trips":            load_trips,
        "drivers":          load_drivers,
        "driver_goals":     load_driver_goals,
        "earnings_velocity":load_earnings_velocity,
    }

    for name, path in paths.items():
        if not path.exists():
            logger.error(f"File not found: {path}")
            datasets[name] = pd.DataFrame()
            continue
        try:
            datasets[name] = loaders[name](str(path))
        except Exception as e:
            logger.error(f"Failed to load {name}: {e}")
            datasets[name] = pd.DataFrame()

    return datasets

if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "./driver_pulse_hackathon_data"
    datasets = load_all(data_dir)
    for name, df in datasets.items():
        print(f"  {name:25s}: {len(df):>5} rows | {list(df.columns)}")
