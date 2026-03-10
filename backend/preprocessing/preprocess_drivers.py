"""
preprocess_drivers.py
---------------------
Cleans and prepares the drivers dataset.

Steps:
- Schema validation
- Remove duplicates
- Handle missing values
- Standardize text
- Fix datatypes
- Validate value ranges
- Save cleaned dataset
"""

import sys
from pathlib import Path
import pandas as pd


# -----------------------------------------------------
# FIX MODULE PATH (so imports work anywhere)
# -----------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))


# -----------------------------------------------------
# MAIN PREPROCESS FUNCTION
# -----------------------------------------------------

def preprocess_drivers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean drivers dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Raw drivers dataset

    Returns
    -------
    pd.DataFrame
        Cleaned drivers dataset
    """

    print("\nStarting drivers preprocessing...")
    print("Initial dataset shape:", df.shape)

    # -------------------------------------------------
    # SCHEMA VALIDATION
    # -------------------------------------------------

    required_columns = [
        "driver_id",
        "name",
        "city",
        "shift_preference",
        "avg_hours_per_day",
        "avg_earnings_per_hour",
        "experience_months",
        "rating"
    ]

    missing_cols = [c for c in required_columns if c not in df.columns]

    if missing_cols:
        raise ValueError(f"Dataset missing required columns: {missing_cols}")

    print("Schema validation passed")

    # -------------------------------------------------
    # REMOVE DUPLICATES
    # -------------------------------------------------

    duplicates = df.duplicated(subset=["driver_id"]).sum()

    if duplicates > 0:
        print("Removing duplicate drivers:", duplicates)
        df = df.drop_duplicates(subset=["driver_id"])
    else:
        print("No duplicate drivers found")

    # -------------------------------------------------
    # FIX NAME COLUMN
    # -------------------------------------------------

    df["name"] = df["name"].fillna(df["driver_id"])
    df.loc[df["name"].astype(str).str.strip() == "", "name"] = df["driver_id"]

    # -------------------------------------------------
    # HANDLE MISSING VALUES
    # -------------------------------------------------

    # categorical columns
    for col in ["city", "shift_preference"]:

        if col in df.columns and df[col].isna().any():

            mode_val = df[col].mode()[0]

            print(f"Filling missing {col} with mode:", mode_val)

            df[col] = df[col].fillna(mode_val)

    # numeric columns
    numeric_cols = [
        "avg_hours_per_day",
        "avg_earnings_per_hour",
        "experience_months",
        "rating"
    ]

    for col in numeric_cols:

        if col in df.columns and df[col].isna().any():

            mean_val = df[col].mean()

            print(f"Filling missing {col} with mean:", round(mean_val, 2))

            df[col] = df[col].fillna(mean_val)

    # -------------------------------------------------
    # STANDARDIZE TEXT
    # -------------------------------------------------

    if "city" in df.columns:
        df["city"] = df["city"].astype(str).str.strip().str.lower()

    if "shift_preference" in df.columns:
        df["shift_preference"] = df["shift_preference"].astype(str).str.strip().str.lower()

    # -------------------------------------------------
    # FIX DATA TYPES
    # -------------------------------------------------

    if "avg_hours_per_day" in df.columns:
        df["avg_hours_per_day"] = df["avg_hours_per_day"].astype(float)

    if "avg_earnings_per_hour" in df.columns:
        df["avg_earnings_per_hour"] = df["avg_earnings_per_hour"].astype(float)

    if "rating" in df.columns:
        df["rating"] = df["rating"].round().astype(int)

    if "experience_months" in df.columns:
        df["experience_months"] = df["experience_months"].round().astype(int)

    # -------------------------------------------------
    # VALIDATE RANGES
    # -------------------------------------------------

    if "rating" in df.columns:
        df["rating"] = df["rating"].clip(0, 5)

    if "avg_hours_per_day" in df.columns:
        df["avg_hours_per_day"] = df["avg_hours_per_day"].clip(0, 24)

    if "experience_months" in df.columns:
        df.loc[df["experience_months"] < 0, "experience_months"] = 0

    # -------------------------------------------------
    # SORT DATA
    # -------------------------------------------------

    if "driver_id" in df.columns:
        df = df.sort_values("driver_id").reset_index(drop=True)

    # -------------------------------------------------
    # DATA HEALTH REPORT
    # -------------------------------------------------

    print("\nDriver Dataset Health Report")

    print("Total drivers:", len(df))

    if "city" in df.columns:
        print("Unique cities:", df["city"].nunique())

    if "shift_preference" in df.columns:
        print("Shift types:", df["shift_preference"].unique())

    if "rating" in df.columns:
        print("Average rating:", round(df["rating"].mean(), 2))

    print("\nDrivers preprocessing completed successfully.")

    return df


# -----------------------------------------------------
# SAVE FUNCTION
# -----------------------------------------------------

def save_cleaned_drivers(df: pd.DataFrame, output_path: Path):

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

    drivers_df = datasets["drivers"]

    cleaned_df = preprocess_drivers(drivers_df)

    save_path = BASE_DIR / "processed_outputs" / "cleaned_drivers.csv"

    save_cleaned_drivers(cleaned_df, save_path)