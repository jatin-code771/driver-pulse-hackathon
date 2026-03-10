# Driver Pulse: Team jatin-code771

Demo Video: [Google Drive Walkthrough](https://drive.google.com/file/d/1yxC8dLirpte5BfJZuqGtbynDpm-iKUoh/view?usp=sharing)

Live Application: [https://driver-pulse-hackathon.vercel.app/](https://driver-pulse-hackathon.vercel.app/)

(Optional) Judge Login Credentials: Username: judge@uber.com | Password: hackathon2026

Note to Judges: The Render backend may take 60 seconds to wake up on first request — please refresh the page if data does not load immediately.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Live Deployment](#live-deployment)
4. [Demo Video](#demo-video)
5. [Setup Instructions](#setup-instructions)
6. [API Reference](#api-reference)
7. [Data Pipeline](#data-pipeline)
8. [Trade-offs & Assumptions](#trade-offs--assumptions)
9. [Signal-to-Event Mapping](#signal-to-event-mapping)

---

## Project Overview

Driver Pulse is a real-time driver analytics dashboard that ingests multi-modal sensor data (accelerometer, audio intensity) along with trip records and earnings logs to surface actionable insights. It detects stress and conflict moments, tracks earnings velocity, and predicts driver behavior — all presented in a clean, interactive React frontend.

Key capabilities:
- **Flagged Moment Detection** — identifies harsh braking, rapid acceleration, and high-noise cabin events linked to specific timestamps and trip IDs.
- **Earnings Velocity Tracking** — overlays real-time earnings checkpoints against driver daily goals.
- **Driver Safety Analytics** — aggregates per-driver safety scores and trend charts.
- **Trip Overview** — searchable trip table with quality ratings and summary statistics.

---

## Architecture

```
driver-pulse-hackathon/
├── backend/                        # Python / Flask API
│   ├── app.py                      # Minimal Flask entry-point (dev)
│   ├── api.py                      # Full API with all routes (production)
│   ├── requirements.txt
│   ├── pipelines/                  # ETL — merges raw sensor + trip data
│   ├── analytics_engine/           # Flagged-moment detection logic
│   ├── prediction_engine/          # Realtime driver prediction models
│   ├── preprocessing/              # Data cleaning & normalisation
│   ├── output_data/                # Processed CSVs served by the API
│   └── driver_pulse_hackathon_data/# Raw hackathon dataset
│       ├── drivers/
│       ├── trips/
│       ├── sensor_data/
│       └── earnings/
│
└── frontend/                       # React 19 + Vite + Tailwind CSS
    └── src/
        ├── App.jsx
        ├── Components/
        │   ├── DriverSafetyAnalytics.jsx
        │   ├── DynamicVelocityChart.jsx
        │   └── FlaggedMoments.jsx
        └── Pages/
            └── TripOverview.jsx
```

**Data flow:**
`Raw CSVs → preprocessing → analytics_engine / prediction_engine → output_data/ → Flask API → React dashboard`

---

## Live Deployment

| Layer | URL |
|-------|-----|
| Frontend (Vercel) | [https://driver-pulse-hackathon.vercel.app/](https://driver-pulse-hackathon.vercel.app/) |
| Backend (Render) | Hosted on Render — see Note to Judges above regarding cold-start delay |

---

## Demo Video

[▶ Watch the 2–3 minute walkthrough on Google Drive](https://drive.google.com/file/d/1yxC8dLirpte5BfJZuqGtbynDpm-iKUoh/view?usp=sharing)

The video covers:
1. Live dashboard tour — flagged moments, earnings velocity, safety analytics
2. Filtering by driver ID and trip ID
3. Brief explanation of the data pipeline and event-detection logic

---

## Setup Instructions

### Prerequisites

- Python 3.10+
- Node.js 18+ and npm

### 1 — Clone the repository

```bash
git clone https://github.com/jatin-code771/driver-pulse-hackathon.git
cd driver-pulse-hackathon
```

### 2 — Backend

```bash
cd backend

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the Flask development server (port 5000)
python api.py
```

The API will be available at `http://localhost:5000`.

> **Note:** The `output_data/` directory must be populated before the API can serve data.  
> If it is empty, run the data pipeline first:
> ```bash
> python pipelines/driver_data_merger.py
> ```

### 3 — Frontend

```bash
cd frontend

# Install dependencies
npm install

# Start the Vite development server (port 5173)
npm run dev
```

Open [http://localhost:5173](http://localhost:5173) in your browser.

> By default the frontend points to the hosted Render backend. To use your local backend, update the API base URL in `src/` (search for the `VITE_API_URL` environment variable or hardcoded base URL) to `http://localhost:5000`.

### 4 — Production build (optional)

```bash
cd frontend
npm run build       # outputs to frontend/dist/
npm run preview     # serves the production build locally
```

---

## API Reference

All endpoints are served from the Flask backend.

| Method | Endpoint | Query Params | Description |
|--------|----------|--------------|-------------|
| GET | `/api/health` | — | Health check |
| GET | `/api/flagged-moments` | `driver_id` | Latest flagged stress/conflict moments |
| GET | `/api/flagged-moments-all` | `driver_id` | Full historical flagged moments |
| GET | `/api/accelerometer` | `trip_id` | Raw accelerometer sensor readings |
| GET | `/api/trips` | `driver_id` | Completed trip records with earnings |
| GET | `/api/driver-predictions` | `driver_id` | Realtime driver behavior predictions |
| GET | `/api/trip-summary` | `driver_id` | Trip summaries with quality ratings |

---

## Data Pipeline

Raw data lives in `backend/driver_pulse_hackathon_data/` and consists of:

| File | Records | Description |
|------|---------|-------------|
| `drivers/drivers.csv` | 210 | Driver profiles |
| `trips/trips.csv` | 220 | Completed trips with earnings |
| `sensor_data/accelerometer_data.csv` | 243 | Motion-pattern readings |
| `sensor_data/audio_intensity_data.csv` | 248 | Cabin noise measurements |
| `earnings/driver_goals.csv` | 210 | Daily earnings goals |
| `earnings/earnings_velocity_log.csv` | 221 | Real-time earnings checkpoints |
| `processed_outputs/flagged_moments.csv` | 208 | Pre-detected stress/conflict moments |
| `processed_outputs/trip_summaries.csv` | 220 | Trip summaries with quality ratings |

The pipeline (`pipelines/driver_data_merger.py`) joins these sources and writes enriched CSVs to `backend/output_data/` which the API reads directly.

---

## Trade-offs & Assumptions

### Speed vs. Precision

| Decision | Trade-off |
|----------|-----------|
| Serving pre-processed static CSVs instead of a database | Eliminates DB setup overhead — fast to deploy but not suitable for live streaming ingestion at scale. |
| Rule-based flagged-moment detection (threshold on accelerometer magnitude + audio level) | Avoids model training time; recall may be lower than an ML classifier trained on labelled events. |
| No authentication on API endpoints | Acceptable for a hackathon demo; would need OAuth / JWT in production. |
| Frontend fetches all records and filters client-side | Simple to implement; would need server-side pagination for datasets larger than ~10 k rows. |

### Assumptions About Noisy Data

- **Accelerometer outliers** — readings more than 3 standard deviations from the per-trip mean were treated as sensor glitches and dropped before computing safety scores.
- **Audio intensity spikes** — a rolling 5-sample median filter was applied to smooth transient noise bursts that do not represent genuine cabin events.
- **Missing `driver_id` values** — rows with null or empty `driver_id` were excluded from all aggregations to avoid polluting per-driver metrics.
- **Timestamp alignment** — trip start/end timestamps are used as the anchor; sensor records are assigned to the nearest trip window using an ±30-second tolerance.
- **Earnings currency** — all monetary values are assumed to be in USD; no currency conversion was performed.

---

## Signal-to-Event Mapping

The table below documents how raw sensor signals are translated into labelled events, enabling reproducibility and auditing of flagged moments.

| Raw Signal | Threshold / Rule | Event Label | Timestamp Source |
|------------|-----------------|-------------|-----------------|
| Accelerometer magnitude > 2.5 g | Single reading exceeds threshold | `harsh_braking` or `rapid_acceleration` (sign of z-axis determines direction) | `accelerometer_data.timestamp` |
| Accelerometer magnitude > 1.8 g (sustained ≥ 3 consecutive readings) | Rolling window | `erratic_driving` | First reading in window |
| Audio intensity > 85 dB | Single reading | `high_cabin_noise` | `audio_intensity_data.timestamp` |
| Audio intensity > 75 dB (sustained ≥ 5 consecutive readings) | Rolling window | `prolonged_noise_event` | First reading in window |
| Composite: harsh_braking + high_cabin_noise within 10 s | Co-occurrence window | `conflict_moment` | Earlier of the two timestamps |
| Earnings velocity < 40 % of daily goal at 50 % of shift elapsed | Earnings log check | `low_earnings_alert` | `earnings_velocity_log.checkpoint_time` |

Each flagged moment written to `flagged_moments_latest.csv` includes: `driver_id`, `trip_id`, `timestamp`, `event_label`, `signal_value`, and `severity` (low / medium / high).