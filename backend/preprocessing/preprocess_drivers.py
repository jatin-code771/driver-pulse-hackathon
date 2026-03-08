import pandas as pd
from pathlib import Path

# =========================================================
# PATH CONFIG
# =========================================================

# backend directory
BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = BASE_DIR / "driver_pulse_hackathon_data" / "drivers" / "drivers.csv"

OUTPUT_PATH = BASE_DIR / "processed_outputs" / "cleaned_drivers.csv"

print("Reading dataset from:", INPUT_PATH)


# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv(INPUT_PATH)

print("Initial dataset shape:", df.shape)


# =========================================================
# SCHEMA VALIDATION
# =========================================================

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


# =========================================================
# REMOVE DUPLICATES
# =========================================================

duplicates = df.duplicated(subset=["driver_id"]).sum()

if duplicates > 0:
    print("Removing duplicate drivers:", duplicates)
    df = df.drop_duplicates(subset=["driver_id"])
else:
    print("No duplicate drivers found")


# =========================================================
# FIX NAME COLUMN
# =========================================================

df["name"] = df["name"].fillna(df["driver_id"])
df.loc[df["name"].astype(str).str.strip() == "", "name"] = df["driver_id"]


# =========================================================
# HANDLE MISSING VALUES
# =========================================================

# categorical columns → mode
for col in ["city", "shift_preference"]:

    if df[col].isna().any():

        mode_val = df[col].mode()[0]

        print(f"Filling missing {col} with mode:", mode_val)

        df[col] = df[col].fillna(mode_val)


# numeric columns → mean
numeric_cols = [
    "avg_hours_per_day",
    "avg_earnings_per_hour",
    "experience_months",
    "rating"
]

for col in numeric_cols:

    if df[col].isna().any():

        mean_val = df[col].mean()

        print(f"Filling missing {col} with mean:", round(mean_val, 2))

        df[col] = df[col].fillna(mean_val)


# =========================================================
# STANDARDIZE TEXT
# =========================================================

df["city"] = df["city"].astype(str).str.strip().str.lower()
df["shift_preference"] = df["shift_preference"].astype(str).str.strip().str.lower()


# =========================================================
# FIX DATA TYPES
# =========================================================

df["avg_hours_per_day"] = df["avg_hours_per_day"].astype(float)
df["avg_earnings_per_hour"] = df["avg_earnings_per_hour"].astype(float)

df["rating"] = df["rating"].round().astype(int)
df["experience_months"] = df["experience_months"].round().astype(int)


# =========================================================
# VALIDATE RANGES
# =========================================================

df["rating"] = df["rating"].clip(0, 5)

df["avg_hours_per_day"] = df["avg_hours_per_day"].clip(0, 24)

df.loc[df["experience_months"] < 0, "experience_months"] = 0


# =========================================================
# SORT DATA
# =========================================================

df = df.sort_values("driver_id").reset_index(drop=True)


# =========================================================
# DATA HEALTH REPORT
# =========================================================

print("\nDriver Dataset Health Report")

print("Total drivers:", len(df))
print("Unique cities:", df["city"].nunique())
print("Shift types:", df["shift_preference"].unique())
print("Average rating:", round(df["rating"].mean(), 2))


# =========================================================
# SAVE DATASET
# =========================================================

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(OUTPUT_PATH, index=False)

print("\nClean dataset saved to:", OUTPUT_PATH)
print("Drivers preprocessing completed successfully.")