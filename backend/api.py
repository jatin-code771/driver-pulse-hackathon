import csv
import os
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output_data")

# CSV file paths — all consolidated in backend/output_data/
CSV_FILES = {
    "flagged_moments_latest": os.path.join(OUTPUT_DIR, "flagged_moments_latest.csv"),
    "flagged_moments": os.path.join(OUTPUT_DIR, "flagged_moments.csv"),
    "accelerometer_data": os.path.join(OUTPUT_DIR, "accelerometer_data.csv"),
    "trips": os.path.join(OUTPUT_DIR, "trips.csv"),
    "realtime_driver_predictions": os.path.join(OUTPUT_DIR, "realtime_driver_predictions.csv"),
}


def read_csv_file(filepath):
    """Read a CSV file and return list of dicts."""
    if not os.path.isfile(filepath):
        return None
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [row for row in reader if any(row.values())]


@app.route("/api/flagged-moments", methods=["GET"])
def get_flagged_moments():
    """Return flagged moments (latest). Optional ?driver_id= filter."""
    rows = read_csv_file(CSV_FILES["flagged_moments_latest"])
    if rows is None:
        return jsonify({"error": "File not found"}), 404
    driver_id = request.args.get("driver_id")
    if driver_id:
        rows = [r for r in rows if r.get("driver_id") == driver_id]
    return jsonify(rows)


@app.route("/api/flagged-moments-all", methods=["GET"])
def get_flagged_moments_all():
    """Return all historical flagged moments. Optional ?driver_id= filter."""
    rows = read_csv_file(CSV_FILES["flagged_moments"])
    if rows is None:
        return jsonify({"error": "File not found"}), 404
    driver_id = request.args.get("driver_id")
    if driver_id:
        rows = [r for r in rows if r.get("driver_id") == driver_id]
    return jsonify(rows)


@app.route("/api/accelerometer", methods=["GET"])
def get_accelerometer_data():
    """Return accelerometer sensor data. Optional ?trip_id= filter."""
    rows = read_csv_file(CSV_FILES["accelerometer_data"])
    if rows is None:
        return jsonify({"error": "File not found"}), 404
    trip_id = request.args.get("trip_id")
    if trip_id:
        rows = [r for r in rows if r.get("trip_id") == trip_id]
    return jsonify(rows)


@app.route("/api/trips", methods=["GET"])
def get_trips():
    """Return trip data. Optional ?driver_id= filter."""
    rows = read_csv_file(CSV_FILES["trips"])
    if rows is None:
        return jsonify({"error": "File not found"}), 404
    driver_id = request.args.get("driver_id")
    if driver_id:
        rows = [r for r in rows if r.get("driver_id") == driver_id]
    return jsonify(rows)


@app.route("/api/driver-predictions", methods=["GET"])
def get_driver_predictions():
    """Return realtime driver predictions. Optional ?driver_id= filter."""
    rows = read_csv_file(CSV_FILES["realtime_driver_predictions"])
    if rows is None:
        return jsonify({"error": "File not found"}), 404
    driver_id = request.args.get("driver_id")
    if driver_id:
        rows = [r for r in rows if r.get("driver_id") == driver_id]
    return jsonify(rows)


@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=5000)
