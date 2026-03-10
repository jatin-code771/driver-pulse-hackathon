"""
preprocess_earnings_velocity_log.py
-----------------------------------
Cleans and prepares the earnings_velocity_log dataset.

Steps:
- Schema validation
- Remove duplicates
- Robust date cleaning
- Timestamp parsing
- Numeric type conversion
- Negative value correction
- Velocity validation
- Forecast status recomputation
- Sorting and integrity checks
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np


# -----------------------------------------------------
# FIX MODULE PATH
# -----------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))


# -----------------------------------------------------
# MAIN PREPROCESS FUNCTION
# -----------------------------------------------------

def preprocess_velocity_log(df: pd.DataFrame) -> pd.DataFrame:

    print("\nStarting earnings velocity preprocessing...")
    print("Dataset shape:", df.shape)

    # -------------------------------------------------
    # SCHEMA VALIDATION
    # -------------------------------------------------

    required_columns = [
        "log_id",
        "driver_id",
        "date",
        "timestamp",
        "cumulative_earnings",
        "elapsed_hours",
        "current_velocity",
        "target_velocity",
        "velocity_delta",
        "trips_completed",
        "forecast_status"
    ]

    missing_cols = [c for c in required_columns if c not in df.columns]

    if missing_cols:
        raise ValueError(f"Dataset missing columns: {missing_cols}")

    print("Schema validation passed.")

    # -------------------------------------------------
    # REMOVE DUPLICATES
    # -------------------------------------------------

    duplicates = df.duplicated(subset=["log_id"]).sum()

    if duplicates > 0:
        print("Removing duplicate logs:", duplicates)
        df = df.drop_duplicates(subset=["log_id"])
    else:
        print("No duplicate logs found.")

    # -------------------------------------------------
    # ROBUST DATE CLEANING
    # -------------------------------------------------

    def extract_year(date_value):

        try:
            year = str(date_value).split("-")[0]
            year = ''.join(filter(str.isdigit, year))

            if len(year) >= 4:
                year = int(year[-4:])
            else:
                return None

            if 2000 <= year <= 2100:
                return year

            return None

        except:
            return None

    df["extracted_year"] = df["date"].apply(extract_year)

    mode_year = df["extracted_year"].dropna().mode()

    if len(mode_year) > 0:
        mode_year = int(mode_year[0])
    else:
        mode_year = 2024

    print("Most common year detected:", mode_year)

    def fix_date(date_value):

        try:

            parts = str(date_value).split("-")

            if len(parts) != 3:
                return f"{mode_year}-02-06"

            year, month, day = parts

            year_digits = ''.join(filter(str.isdigit, year))

            if len(year_digits) >= 4:
                year_digits = int(year_digits[-4:])
            else:
                year_digits = mode_year

            if year_digits < 2000 or year_digits > 2100:
                year_digits = mode_year

            return f"{year_digits:04d}-{month}-{day}"

        except:

            return f"{mode_year}-02-06"

    df["date"] = df["date"].apply(fix_date)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df.drop(columns=["extracted_year"], inplace=True)

    # -------------------------------------------------
    # FIX TIMESTAMP
    # -------------------------------------------------

    print("Processing timestamps...")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce").dt.time

    invalid_ts = df["timestamp"].isna().sum()

    print("Invalid timestamps:", invalid_ts)

    # -------------------------------------------------
    # NUMERIC TYPE CONVERSION
    # -------------------------------------------------

    numeric_cols = [
        "cumulative_earnings",
        "elapsed_hours",
        "current_velocity",
        "target_velocity",
        "velocity_delta",
        "trips_completed"
    ]

    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    # -------------------------------------------------
    # NEGATIVE VALUE CHECK
    # -------------------------------------------------

    for col in ["cumulative_earnings", "elapsed_hours", "current_velocity"]:

        negatives = (df[col] < 0).sum()

        if negatives > 0:

            print(f"Fixing negative values in {col}:", negatives)

            df.loc[df[col] < 0, col] = 0

    # -------------------------------------------------
    # VALIDATE VELOCITY FORMULA
    # -------------------------------------------------

    df["calculated_velocity"] = (
        df["cumulative_earnings"] /
        df["elapsed_hours"].replace(0, np.nan)
    )

    tolerance = 0.01

    incorrect_velocity = abs(df["current_velocity"] - df["calculated_velocity"]) > tolerance

    print("Velocity mismatches:", incorrect_velocity.sum())

    df.loc[incorrect_velocity, "current_velocity"] = df.loc[
        incorrect_velocity, "calculated_velocity"
    ]

    df.drop(columns=["calculated_velocity"], inplace=True)

    # -------------------------------------------------
    # RECOMPUTE VELOCITY DELTA
    # -------------------------------------------------

    df["velocity_delta"] = df["current_velocity"] - df["target_velocity"]

    df["velocity_delta"] = df["velocity_delta"].round(2)
    df["current_velocity"] = df["current_velocity"].round(2)

    # -------------------------------------------------
    # FIX FORECAST STATUS
    # -------------------------------------------------

    def forecast_logic(delta):

        if delta > 5:
            return "ahead"

        elif delta < -5:
            return "at_risk"

        else:
            return "on_track"

    df["forecast_status"] = df["velocity_delta"].apply(forecast_logic)

    df.loc[df["trips_completed"] < 0, "trips_completed"] = 0

    df["trips_completed"] = df["trips_completed"].astype(int)

    # -------------------------------------------------
    # SORT DATA
    # -------------------------------------------------

    df = df.sort_values(["driver_id", "timestamp"])

    overlap = df["timestamp"] < df.groupby("driver_id")["timestamp"].shift()

    print("Out of order logs:", overlap.sum())

    # -------------------------------------------------
    # DATA HEALTH REPORT
    # -------------------------------------------------

    print("\nDataset Health Report")

    print("Total logs:", len(df))
    print("Unique drivers:", df["driver_id"].nunique())
    print("Average velocity:", round(df["current_velocity"].mean(), 2))
    print("Drivers ahead:", (df["forecast_status"] == "ahead").sum())
    print("Drivers at risk:", (df["forecast_status"] == "at_risk").sum())

    print("\nVelocity log preprocessing completed successfully.")

    return df


# -----------------------------------------------------
# SAVE FUNCTION
# -----------------------------------------------------

def save_cleaned_velocity_log(df: pd.DataFrame, output_path: Path):

    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)

    print("\nClean dataset saved to:", output_path)


# -----------------------------------------------------
# RUN FILE DIRECTLY (TEST MODE)
# -----------------------------------------------------

if __name__ == "__main__":

    from jatin.data_ingestion import load_all

    BASE_DIR = Path(__file__).resolve().parent.parent

    datasets = load_all(BASE_DIR / "driver_pulse_hackathon_data")

    velocity_df = datasets["earnings_velocity"]

    cleaned_df = preprocess_velocity_log(velocity_df)

    save_path = BASE_DIR / "processed_outputs" / "cleaned_velocity_log.csv"

    save_cleaned_velocity_log(cleaned_df, save_path)