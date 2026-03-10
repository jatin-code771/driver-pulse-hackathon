"""
preprocess_driver_goals.py
--------------------------
Cleans and prepares the driver_goals dataset.

Steps performed:
- Remove duplicates
- Handle missing values
- Fix negative values
- Convert datatypes
- Validate earnings velocity
- Standardize column names
- Save cleaned dataset
"""
import sys
from pathlib import Path

# Fix module path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

import pandas as pd
from jatin.data_ingestion import load_all

# ---------------------------------------------------
# MAIN PREPROCESSING FUNCTION
# ---------------------------------------------------

def preprocess_driver_goals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the driver_goals dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Raw driver_goals dataset

    Returns
    -------
    pd.DataFrame
        Cleaned dataset
    """

    print("\nStarting driver goals preprocessing...")
    print("Initial dataset shape:", df.shape)

    # ---------------------------------------------------
    # REMOVE DUPLICATES
    # ---------------------------------------------------

    duplicate_rows = df.duplicated().sum()
    print("Duplicate rows found:", duplicate_rows)

    if duplicate_rows > 0:
        df = df.drop_duplicates()

    # ---------------------------------------------------
    # HANDLE MISSING VALUES
    # ---------------------------------------------------

    print("\nMissing values per column:")
    print(df.isnull().sum())

    # Fill date with most common value
    if "date" in df.columns and df["date"].isnull().sum() > 0:
        most_common_date = df["date"].mode()[0]
        df["date"] = df["date"].fillna(most_common_date)

    numeric_cols = [
        "target_earnings",
        "target_hours",
        "current_earnings",
        "current_hours",
        "earnings_velocity"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Categorical columns
    if "status" in df.columns:
        df["status"] = df["status"].fillna(df["status"].mode()[0])

    if "goal_completion_forecast" in df.columns:
        df["goal_completion_forecast"] = df[
            "goal_completion_forecast"
        ].fillna(df["goal_completion_forecast"].mode()[0])

    # ---------------------------------------------------
    # FIX NEGATIVE VALUES
    # ---------------------------------------------------

    for col in numeric_cols:

        if col in df.columns:

            negatives = (df[col] < 0).sum()

            if negatives > 0:
                print(f"Fixing {negatives} negative values in column: {col}")
                df.loc[df[col] < 0, col] = 0

    # ---------------------------------------------------
    # DATATYPE CONVERSION
    # ---------------------------------------------------

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    if "shift_start_time" in df.columns:
        df["shift_start_time"] = pd.to_datetime(
            df["shift_start_time"], errors="coerce"
        ).dt.time

    if "shift_end_time" in df.columns:
        df["shift_end_time"] = pd.to_datetime(
            df["shift_end_time"], errors="coerce"
        ).dt.time

    # ---------------------------------------------------
    # VALIDATE EARNINGS VELOCITY
    # ---------------------------------------------------

    if {"current_earnings", "current_hours", "earnings_velocity"}.issubset(df.columns):

        safe_hours = df["current_hours"].replace(0, pd.NA)

        df["calculated_velocity"] = df["current_earnings"] / safe_hours

        velocity_error = abs(
            df["earnings_velocity"] - df["calculated_velocity"]
        ) > 0.01

        print("Rows with incorrect velocity:", velocity_error.sum())

        df.loc[velocity_error, "earnings_velocity"] = df.loc[
            velocity_error, "calculated_velocity"
        ]

        df.drop(columns=["calculated_velocity"], inplace=True)

    # ---------------------------------------------------
    # RENAME COLUMNS FOR CLARITY
    # ---------------------------------------------------

    rename_map = {
        "status": "dataset_status",
        "goal_completion_forecast": "dataset_forecast"
    }

    df.rename(columns=rename_map, inplace=True)

    # Remove redundant velocity column if needed
    if "earnings_velocity" in df.columns:
        df.drop(columns=["earnings_velocity"], inplace=True)

    # ---------------------------------------------------
    # FINAL COLUMN ORDER
    # ---------------------------------------------------

    desired_columns = [
        "goal_id",
        "driver_id",
        "date",
        "shift_start_time",
        "shift_end_time",
        "target_earnings",
        "target_hours",
        "current_earnings",
        "current_hours",
        "dataset_status",
        "dataset_forecast"
    ]

    existing_cols = [c for c in desired_columns if c in df.columns]

    df = df[existing_cols]

    print("\nFinal dataset shape:", df.shape)
    print("Columns:", df.columns.tolist())

    print("\nDriver goals preprocessing completed successfully.")

    return df


# ---------------------------------------------------
# OPTIONAL SAVE FUNCTION
# ---------------------------------------------------

def save_cleaned_driver_goals(df: pd.DataFrame, output_path: Path):

    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)

    print("\nClean dataset saved to:", output_path)


# ---------------------------------------------------
# RUN FILE DIRECTLY (FOR TESTING)
# ---------------------------------------------------

if __name__ == "__main__":

    BASE_DIR = Path(__file__).resolve().parent.parent

    from jatin.data_ingestion import load_all

    datasets = load_all(BASE_DIR / "driver_pulse_hackathon_data")

    driver_goals_df = datasets["driver_goals"]

    cleaned_df = preprocess_driver_goals(driver_goals_df)

    save_path = BASE_DIR / "processed_outputs" / "cleaned_driver_goals.csv"

    save_cleaned_driver_goals(cleaned_df, save_path)