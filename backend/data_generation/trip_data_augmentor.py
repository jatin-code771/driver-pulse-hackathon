import pandas as pd
import numpy as np
import uuid
from datetime import timedelta
from pathlib import Path

# ---------------------------------------------------
# PATHS
# ---------------------------------------------------

# backend folder
BASE_DIR = Path(__file__).resolve().parent.parent

input_path = BASE_DIR / "driver_pulse_hackathon_data" / "trips" / "trips.csv"

print("Reading dataset from:", input_path)

df = pd.read_csv(input_path)

# ---------------------------------------------------
# FIX TIME COLUMNS
# ---------------------------------------------------

df["start_time"] = pd.to_datetime(df["date"] + " " + df["start_time"], errors="coerce")
df["end_time"] = pd.to_datetime(df["date"] + " " + df["end_time"], errors="coerce")

# ---------------------------------------------------
# LOCATIONS
# ---------------------------------------------------

locations = [
"Andheri West","BKC","Lower Parel","Worli","Bandra","Powai",
"Goregaon","Malad","Whitefield","Indiranagar","Koramangala",
"Electronic City","Dwarka","Connaught Place","Cyber City",
"South Delhi","Kharadi","Viman Nagar","Hinjewadi",
"Banjara Hills","Hitec City","Secunderabad","HSR Layout",
"Marathahalli","Jayanagar","Colaba"
]

# ---------------------------------------------------
# GENERATE EXTRA TRIPS
# ---------------------------------------------------

new_rows = []

for driver in df["driver_id"].unique():

    driver_df = df[df["driver_id"] == driver].sort_values("start_time")

    target_trips = 3
    if driver == "DRV004":
        target_trips = 6

    current_trips = len(driver_df)

    if current_trips >= target_trips:
        continue

    last_trip = driver_df.iloc[-1]

    last_start = last_trip["start_time"]

    for _ in range(target_trips - current_trips):

        start = last_start + timedelta(minutes=np.random.randint(20, 60))

        duration = np.random.randint(18, 35)

        end = start + timedelta(minutes=duration)

        distance = round(np.random.uniform(8, 18), 1)

        surge = round(np.random.uniform(1.0, 1.6), 1)

        fare = int(distance * 15 * surge)

        pickup = np.random.choice(locations)
        drop = np.random.choice(locations)

        new_rows.append({
            "trip_id": "TRIP" + str(uuid.uuid4())[:6],
            "driver_id": driver,
            "date": start.date(),
            "start_time": start,
            "end_time": end,
            "duration_min": duration,
            "distance_km": distance,
            "fare": fare,
            "surge_multiplier": surge,
            "pickup_location": pickup,
            "dropoff_location": drop,
            "trip_status": "completed"
        })

        last_start = start

# ---------------------------------------------------
# MERGE DATA
# ---------------------------------------------------

extra_df = pd.DataFrame(new_rows)

final_df = pd.concat([df, extra_df])

final_df = final_df.sort_values(["driver_id", "start_time"])

# ---------------------------------------------------
# CONVERT BACK TO TIME FORMAT
# ---------------------------------------------------

final_df["start_time"] = final_df["start_time"].dt.time
final_df["end_time"] = final_df["end_time"].dt.time

# ---------------------------------------------------
# SAVE BACK TO SAME FILE
# ---------------------------------------------------

final_df.to_csv(input_path, index=False)

print("✅ Trips dataset expanded and saved.")
print("📁 Saved to:", input_path)