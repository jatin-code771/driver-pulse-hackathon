"""
preprocess_trips.py
-------------------
Cleans and prepares the trips dataset.

Steps:
- Schema validation
- Remove duplicates
- Handle missing trip IDs
- Fix missing timestamps
- Handle missing distance & fare
- Feature engineering
- Data integrity checks
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

def preprocess_trips(df: pd.DataFrame) -> pd.DataFrame:

    print("\nStarting trips preprocessing...")
    print("Initial dataset shape:", df.shape)

    # -------------------------------------------------
    # SCHEMA VALIDATION
    # -------------------------------------------------

    required_columns = [
        "trip_id","driver_id","date","start_time","end_time",
        "duration_min","distance_km","fare","surge_multiplier",
        "pickup_location","dropoff_location","trip_status"
    ]

    missing_cols = [c for c in required_columns if c not in df.columns]

    if missing_cols:
        raise ValueError(f"Dataset missing required columns: {missing_cols}")

    print("Schema validation passed")

    # -------------------------------------------------
    # REMOVE DUPLICATES
    # -------------------------------------------------

    df = df.drop_duplicates(subset=["trip_id"])

    # -------------------------------------------------
    # HANDLE MISSING TRIP IDS
    # -------------------------------------------------

    def fill_trip_ids(group):

        group = group.sort_values(by=["date","start_time"], na_position="last")

        trip_numbers = (
            group["trip_id"]
            .dropna()
            .str.extract(r'(\d+)')
            .astype(float)
        )

        max_id = trip_numbers.max().values[0] if not trip_numbers.empty else 0

        counter = int(max_id)

        for idx,row in group.iterrows():

            if pd.isna(row["trip_id"]):

                counter += 1
                group.at[idx,"trip_id"] = f"TRIP{str(counter).zfill(4)}"

        return group


    if df["trip_id"].isna().any():

        print("Missing trip_ids detected")

        df = df.groupby("driver_id", group_keys=False).apply(fill_trip_ids)

    # -------------------------------------------------
    # HANDLE TEXT MISSING VALUES
    # -------------------------------------------------

    text_cols = ["pickup_location","dropoff_location","trip_status"]

    for col in text_cols:

        if df[col].isna().any():

            mode_val = df[col].mode()[0]

            print(f"Filling missing {col} with mode:", mode_val)

            df[col] = df[col].fillna(mode_val)

    # -------------------------------------------------
    # SURGE MULTIPLIER
    # -------------------------------------------------

    df["surge_multiplier"] = df["surge_multiplier"].fillna(1)

    # -------------------------------------------------
    # CONVERT DATE & TIME
    # -------------------------------------------------

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df["start_time"] = pd.to_datetime(
        df["date"].astype(str) + " " + df["start_time"].astype(str),
        errors="coerce"
    )

    df["end_time"] = pd.to_datetime(
        df["date"].astype(str) + " " + df["end_time"].astype(str),
        errors="coerce"
    )

    # -------------------------------------------------
    # FIX MISSING START / END TIMES
    # -------------------------------------------------

    duration_td = pd.to_timedelta(df["duration_min"], unit="m")

    mask_start_missing = df["start_time"].isna() & df["end_time"].notna()

    df.loc[mask_start_missing,"start_time"] = (
        df.loc[mask_start_missing,"end_time"] - duration_td[mask_start_missing]
    )

    mask_end_missing = df["end_time"].isna() & df["start_time"].notna()

    df.loc[mask_end_missing,"end_time"] = (
        df.loc[mask_end_missing,"start_time"] + duration_td[mask_end_missing]
    )

    # -------------------------------------------------
    # FIX START > END EDGE CASE
    # -------------------------------------------------

    print("Fixing invalid trips")

    df["duration_td"] = pd.to_timedelta(df["duration_min"], unit="m")

    df = df.sort_values(["driver_id","start_time"])

    previous_end = {}

    for i,row in df.iterrows():

        driver = row["driver_id"]

        start = row["start_time"]
        end = row["end_time"]
        duration = row["duration_td"]

        if pd.isna(start) or pd.isna(end):
            continue

        if start > end:

            candidate_start = end - duration

            prev_end = previous_end.get(driver)

            if prev_end is None or candidate_start >= prev_end:
                df.at[i,"start_time"] = candidate_start
            else:
                df.at[i,"end_time"] = start + duration

        previous_end[driver] = df.at[i,"end_time"]

    # -------------------------------------------------
    # HANDLE DISTANCE & FARE MISSING
    # -------------------------------------------------

    valid = df[
        (df["fare"].notna()) &
        (df["distance_km"].notna()) &
        (df["distance_km"] > 0)
    ]

    avg_fare_per_km = (valid["fare"] / valid["distance_km"]).mean()

    route_distance = (
        df.groupby(["pickup_location","dropoff_location"])["distance_km"]
        .mean()
    )

    route_fare = (
        df.groupby(["pickup_location","dropoff_location"])["fare"]
        .mean()
    )

    for idx,row in df[df["distance_km"].isna()].iterrows():

        route = (row["pickup_location"], row["dropoff_location"])

        if route in route_distance:

            df.at[idx,"distance_km"] = route_distance[route]

        elif pd.notna(row["fare"]):

            df.at[idx,"distance_km"] = row["fare"] / avg_fare_per_km


    for idx,row in df[df["fare"].isna()].iterrows():

        route = (row["pickup_location"], row["dropoff_location"])

        if route in route_fare:

            df.at[idx,"fare"] = route_fare[route]

        elif pd.notna(row["distance_km"]):

            df.at[idx,"fare"] = row["distance_km"] * avg_fare_per_km

    # -------------------------------------------------
    # FEATURE ENGINEERING
    # -------------------------------------------------

    df["duration_hours"] = df["duration_min"] / 60

    # -------------------------------------------------
    # DROP REDUNDANT COLUMNS
    # -------------------------------------------------

    df.drop(columns=["duration_min","duration_td"], inplace=True)

    # -------------------------------------------------
    # FIX DATA TYPES
    # -------------------------------------------------

    float_cols = [
        "distance_km",
        "fare",
        "surge_multiplier",
        "duration_hours"
    ]

    df[float_cols] = df[float_cols].astype(float).round(2)

    # -------------------------------------------------
    # FINAL SORT
    # -------------------------------------------------

    df = df.sort_values(["driver_id","start_time"]).reset_index(drop=True)

    # -------------------------------------------------
    # DATA HEALTH REPORT
    # -------------------------------------------------

    print("\nTrips Dataset Health Report")

    print("Total trips:", len(df))
    print("Unique drivers:", df["driver_id"].nunique())

    print("\nTrips preprocessing completed successfully.")

    return df


# -----------------------------------------------------
# SAVE FUNCTION
# -----------------------------------------------------

def save_cleaned_trips(df: pd.DataFrame, output_path: Path):

    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path,index=False)

    print("\nClean dataset saved to:", output_path)


# -----------------------------------------------------
# RUN FILE DIRECTLY (TEST MODE)
# -----------------------------------------------------

if __name__ == "__main__":

    from jatin.data_ingestion import load_all

    BASE_DIR = Path(__file__).resolve().parent.parent

    datasets = load_all(BASE_DIR / "driver_pulse_hackathon_data")

    trips_df = datasets["trips"]

    cleaned_df = preprocess_trips(trips_df)

    save_path = BASE_DIR / "processed_outputs" / "cleaned_trips.csv"

    save_cleaned_trips(cleaned_df, save_path)