import pandas as pd
from pathlib import Path

# ---------------------------------------
# PATH CONFIGURATION
# ---------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

input_path = BASE_DIR / "driver_pulse_hackathon_data" / "earnings" / "driver_goals.csv"

output_path = BASE_DIR / "processed_outputs" / "cleaned_driver_goals.csv"

print("📂 Loading dataset from:", input_path)

df = pd.read_csv(input_path)

print("Initial dataset shape:", df.shape)

# ---------------------------------------
# REMOVE DUPLICATES
# ---------------------------------------

duplicate_rows = df.duplicated().sum()

print("Duplicate rows found:", duplicate_rows)

if duplicate_rows > 0:
    df = df.drop_duplicates()

# ---------------------------------------
# HANDLE MISSING VALUES
# ---------------------------------------

print("\nMissing values per column:")
print(df.isnull().sum())

# Fill date with most common date
if df["date"].isnull().sum() > 0:
    most_common_date = df["date"].mode()[0]
    df["date"] = df["date"].fillna(most_common_date)

# Numeric columns
numeric_cols = [
    "target_earnings",
    "target_hours",
    "current_earnings",
    "current_hours",
    "earnings_velocity"
]

for col in numeric_cols:
    df[col] = df[col].fillna(0)

# Categorical columns
df["status"] = df["status"].fillna(df["status"].mode()[0])
df["goal_completion_forecast"] = df["goal_completion_forecast"].fillna(
    df["goal_completion_forecast"].mode()[0]
)

# ---------------------------------------
# FIX NEGATIVE VALUES
# ---------------------------------------

for col in numeric_cols:

    negatives = (df[col] < 0).sum()

    if negatives > 0:
        print(f"Fixing {negatives} negative values in column: {col}")
        df.loc[df[col] < 0, col] = 0

# ---------------------------------------
# DATATYPE CONVERSION
# ---------------------------------------

df["date"] = pd.to_datetime(df["date"], errors="coerce")

df["shift_start_time"] = pd.to_datetime(
    df["shift_start_time"], errors="coerce"
).dt.time

df["shift_end_time"] = pd.to_datetime(
    df["shift_end_time"], errors="coerce"
).dt.time

# ---------------------------------------
# VALIDATE EARNING VELOCITY
# ---------------------------------------

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

# ---------------------------------------
# RENAME DATASET LABELS
# ---------------------------------------

df.rename(
    columns={
        "status": "dataset_status",
        "goal_completion_forecast": "dataset_forecast"
    },
    inplace=True
)

# Remove redundant velocity column
df.drop(columns=["earnings_velocity"], inplace=True)

# ---------------------------------------
# FINAL COLUMN ORDER
# ---------------------------------------

columns = [
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

df = df[columns]

# ---------------------------------------
# FINAL DATASET INFO
# ---------------------------------------

print("\nFinal dataset shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

# ---------------------------------------
# SAVE CLEAN DATASET
# ---------------------------------------

output_path.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(output_path, index=False)

print("\n✅ Clean dataset saved to:", output_path)
print("🚀 Driver goals preprocessing completed successfully.")