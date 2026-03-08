import pandas as pd
from pathlib import Path

print("Loading datasets...")

# =========================================================
# PATH CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "processed_outputs"

drivers = pd.read_csv(DATA_DIR / "cleaned_drivers.csv")
goals = pd.read_csv(DATA_DIR / "cleaned_driver_goals.csv")
velocity = pd.read_csv(DATA_DIR / "cleaned_velocity_log.csv")
trips = pd.read_csv(DATA_DIR / "cleaned_trips.csv")

print("Datasets loaded")


# =========================================================
# FIX VELOCITY TIMESTAMP
# =========================================================

print("Fixing timestamps...")

velocity["timestamp"] = pd.to_datetime(
    velocity["date"].astype(str) + " " +
    pd.to_datetime(velocity["timestamp"], errors="coerce").dt.strftime("%H:%M:%S"),
    errors="coerce"
)


# =========================================================
# FIX GOAL TIMES
# =========================================================

goals["shift_start_time"] = pd.to_datetime(
    goals["date"].astype(str) + " " + goals["shift_start_time"].astype(str),
    errors="coerce"
)

goals["shift_end_time"] = pd.to_datetime(
    goals["date"].astype(str) + " " + goals["shift_end_time"].astype(str),
    errors="coerce"
)


# =========================================================
# REMOVE OVERLAPPING GOALS
# =========================================================

print("Resolving overlapping goals...")

goals = goals.sort_values(
    ["driver_id", "date", "shift_start_time"]
)

goals = goals.drop_duplicates(
    subset=["driver_id", "date"],
    keep="last"
)


# =========================================================
# CREATE SYNTHETIC GOALS FOR DRIVERS WITHOUT GOALS
# =========================================================

print("Creating synthetic goals if missing...")

velocity["timestamp"] = pd.to_datetime(velocity["timestamp"])

missing_drivers = set(velocity["driver_id"]) - set(goals["driver_id"])

synthetic_goals = []

for drv in missing_drivers:

    drv_logs = velocity[velocity["driver_id"] == drv]

    start_time = drv_logs["timestamp"].min()

    profile = drivers[drivers["driver_id"] == drv].iloc[0]

    avg_hours = profile["avg_hours_per_day"]
    avg_rate = profile["avg_earnings_per_hour"]

    target_earnings = avg_hours * avg_rate

    synthetic_goals.append({
        "goal_id": f"SYN_{drv}",
        "driver_id": drv,
        "date": drv_logs["date"].iloc[0],
        "shift_start_time": start_time,
        "shift_end_time": start_time + pd.Timedelta(hours=avg_hours),
        "target_earnings": target_earnings,
        "target_hours": avg_hours
    })

synthetic_goals = pd.DataFrame(synthetic_goals)

goals = pd.concat([goals, synthetic_goals], ignore_index=True)


# =========================================================
# MERGE DRIVER PROFILE
# =========================================================

print("Merging drivers...")

df = velocity.merge(
    drivers,
    on="driver_id",
    how="left"
)


# =========================================================
# MERGE GOALS
# =========================================================

print("Merging goals...")

df = df.merge(
    goals,
    on=["driver_id", "date"],
    how="left"
)


# =========================================================
# SORT DRIVER TIMELINE
# =========================================================

df = df.sort_values(
    ["driver_id", "timestamp"]
).reset_index(drop=True)


# =========================================================
# FIX CUMULATIVE EARNINGS CONSISTENCY
# =========================================================

print("Fixing cumulative earnings consistency...")

df["cumulative_earnings"] = df.groupby("driver_id")["cumulative_earnings"].cummax()


# =========================================================
# FEATURE ENGINEERING
# =========================================================

print("Calculating metrics...")

df["remaining_earnings"] = (
    df["target_earnings"] - df["cumulative_earnings"]
).clip(lower=0)

df["goal_progress_percent"] = (
    df["cumulative_earnings"] /
    df["target_earnings"].replace(0, pd.NA)
) * 100

df["goal_progress_percent"] = df["goal_progress_percent"].clip(upper=100)

df["remaining_shift_hours"] = (
    (df["shift_end_time"] - df["timestamp"])
    .dt.total_seconds() / 3600
).clip(lower=0)

df["estimated_hours_to_goal"] = (
    df["remaining_earnings"] /
    df["current_velocity"].replace(0, pd.NA)
)

df["overtime_required"] = (
    df["estimated_hours_to_goal"] > df["remaining_shift_hours"]
)


# =========================================================
# DROP UNUSED COLUMNS
# =========================================================

df = df.drop(
    columns=["current_earnings", "current_hours"],
    errors="ignore"
)


# =========================================================
# FINAL SORT
# =========================================================

df = df.sort_values(
    ["driver_id", "timestamp"]
)


# =========================================================
# SAVE MASTER DATASET
# =========================================================

print("Saving dataset...")

output = DATA_DIR / "final_driver_timeline.csv"

df.to_csv(output, index=False)

print("SUCCESS")
print("Rows:", len(df))
print("Saved to:", output)