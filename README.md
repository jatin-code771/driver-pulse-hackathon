# 🚗 Driver Pulse

[![Live Demo](https://img.shields.io/badge/Live%20Demo-driver--pulse--hackathon.vercel.app-brightgreen?style=for-the-badge&logo=vercel)](https://driver-pulse-hackathon.vercel.app)
[![Python](https://img.shields.io/badge/Python-68%25-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![JavaScript](https://img.shields.io/badge/JavaScript-24.4%25-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black)](https://developer.mozilla.org/en-US/docs/Web/JavaScript)
[![CSS](https://img.shields.io/badge/CSS-7.3%25-1572B6?style=for-the-badge&logo=css3&logoColor=white)](https://developer.mozilla.org/en-US/docs/Web/CSS)

> **Real-time driver behavior analytics and earnings prediction for ride-hailing platforms.**

Driver Pulse is a full-stack hackathon project that ingests sensor data (accelerometer, audio) from driver trips, processes it through an analytics engine to detect harsh driving and noise events, and surfaces real-time earnings-velocity predictions through an interactive dashboard.

---

## 🌐 Live Demo

👉 **[https://driver-pulse-hackathon.vercel.app](https://driver-pulse-hackathon.vercel.app)**

---

## ✨ Features

- 📊 **Real-time earnings velocity** — live chart of driver pacing vs. goal progress
- 🚨 **Flagged driving moments** — automatic detection of harsh braking, acceleration, swerving, noise spikes, and argument signals
- 🧠 **Prediction engine** — real-time driver goal predictions with cold-start protection and velocity capping
- 🔍 **Safety analytics dashboard** — per-driver safety scores and trend charts
- 🗂️ **Trip overview** — filterable trip data with fare, distance, and timestamp details
- 🐳 **Dockerized backend** — easy self-hosted deployment

---

## 🛠️ Tech Stack

| Layer | Technologies |
|-------|-------------|
| **Backend** | Python 3.11, Flask, pandas, numpy |
| **Frontend** | React 19, Vite 8, Tailwind CSS, Recharts, PapaParse, react-icons |
| **Deployment** | Docker (backend), Vercel (frontend) |

---

## 📁 Project Structure

```
driver-pulse-hackathon/
├── backend/
│   ├── analytics_engine/        # Core analytics: data ingestion, signal preprocessing,
│   │                            #   feature engineering, output generation,
│   │                            #   decision logging, accuracy comparison
│   ├── data_generation/         # Synthetic data augmentation (velocity_data_augmentor.py)
│   ├── pipelines/               # Data merge pipelines (driver_data_merger.py)
│   ├── prediction_engine/       # Real-time driver goal prediction engine
│   ├── preprocessing/           # Data cleaning scripts for each dataset
│   ├── driver_pulse_hackathon_data/  # Raw hackathon datasets
│   ├── output_data/             # Generated output CSVs
│   ├── processed_outputs/       # Cleaned/processed CSVs
│   ├── driver_outputs/          # Per-driver output files
│   ├── utils/                   # Utility functions
│   ├── api.py                   # Flask REST API (main entry point)
│   ├── app.py                   # Alternative Flask app entry point
│   ├── requirements.txt         # Python dependencies
│   └── Dockerfile               # Docker config (python:3.11-slim, port 5000)
├── frontend/
│   ├── src/
│   │   ├── Components/
│   │   │   ├── DriverSafetyAnalytics.jsx   # Safety analytics dashboard component
│   │   │   ├── DynamicVelocityChart.jsx    # Real-time earnings velocity chart (Recharts)
│   │   │   └── FlaggedMoments.jsx          # Flagged driving moments display
│   │   ├── Pages/
│   │   │   └── TripOverview.jsx            # Main trip overview dashboard page
│   │   ├── styles/                         # CSS styles
│   │   ├── assets/                         # Static assets
│   │   ├── App.jsx                         # Root React component
│   │   └── main.jsx                        # React entry point
│   ├── package.json              # npm dependencies
│   ├── vite.config.js            # Vite config with API proxy to backend (:5000)
│   ├── tailwind.config.js        # Tailwind CSS config
│   └── index.html                # HTML entry point
└── README.md
```

---

## 🔌 API Endpoints

Base URL (local): `http://localhost:5000`

| Method | Endpoint | Description | Query Params |
|--------|----------|-------------|--------------|
| `GET` | `/api/health` | Health check | — |
| `GET` | `/api/flagged-moments` | Latest flagged driving moments | `driver_id` (optional) |
| `GET` | `/api/flagged-moments-all` | All historical flagged moments | `driver_id` (optional) |
| `GET` | `/api/accelerometer` | Accelerometer sensor data | `trip_id` (optional) |
| `GET` | `/api/trips` | Trip data | `driver_id` (optional) |
| `GET` | `/api/driver-predictions` | Real-time driver predictions | `driver_id` (optional) |
| `GET` | `/api/trip-summary` | Trip summary data | `driver_id` (optional) |

---

## 🧠 Analytics Engine

The analytics engine processes raw sensor data through a multi-stage pipeline:

### Accelerometer Processing
- Computes signal **magnitude** from raw XYZ axes
- **Clips outliers** using Z-score threshold of 4.0
- **Smooths signals** with a rolling window of 3 samples
- Computes **deltas** between consecutive samples

### Audio Processing
- Cleans dB levels (valid range: 30–120 dB)
- **Encodes severity** classifications on a quiet→argument scale

### Feature Engineering
| Feature | Detection Logic |
|---------|----------------|
| Harsh braking | Deceleration magnitude > 2.5 g |
| Harsh acceleration | Acceleration magnitude > 2.5 g |
| Lateral swerve | Lateral magnitude threshold crossing |
| Noise spike | Audio level > 85 dB |
| Argument signal | Sustained noise + severity encoding |

### Flagged Moment Types
Each flagged moment includes a **severity score** (low / medium / high):

| Flag | Description |
|------|-------------|
| `conflict_moment` | In-cabin conflict detected via audio |
| `harsh_driving` | Sudden braking or acceleration |
| `noise_event` | Significant noise spike |
| `stress_signal` | Combined stress indicators |

**Decision logging** tracks every threshold crossing with the raw sensor value.

---

## 🔮 Prediction Engine

- Calculates **pacing ratio**, **target velocity**, and **progress percentage** in real time
- **Cold-start protection** for newly onboarded drivers with limited history
- **Earnings velocity capping** to filter out unrealistic spikes

---

## 🧹 Data Preprocessing

Individual preprocessing scripts handle each dataset:

| Script | Responsibilities |
|--------|-----------------|
| `preprocess_drivers.py` | Schema validation, deduplication, missing value imputation, text standardization |
| `preprocess_trips.py` | Trip ID generation, timestamp fixing, fare/distance handling |
| `preprocess_driver_goals.py` | Negative value correction, earnings velocity validation |
| `preprocess_earnings_velocity_log.py` | Robust date cleaning, velocity recomputation |

---

## 🖥️ Frontend Components

| Component | Description |
|-----------|-------------|
| `TripOverview` | Main dashboard page with filterable trip data visualization |
| `DriverSafetyAnalytics` | Safety analytics charts and per-driver metrics |
| `DynamicVelocityChart` | Real-time earnings velocity line chart built with Recharts |
| `FlaggedMoments` | Flagged driving events with color-coded severity indicators |

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.11+
- Node.js 18+ and npm
- Docker (optional, for containerized backend)

### 🐍 Backend (Local)

```bash
cd backend
pip install -r requirements.txt
python api.py
# API available at http://localhost:5000
```

### ⚛️ Frontend (Local)

```bash
cd frontend
npm install
npm run dev
# App available at http://localhost:5173
# API requests are proxied to http://localhost:5000
```

### 🐳 Docker (Backend)

```bash
cd backend
docker build -t driver-pulse-backend .
docker run -p 5000:5000 driver-pulse-backend
# API available at http://localhost:5000
```

---

## 📊 Language Composition

| Language | Percentage |
|----------|-----------|
| 🐍 Python | 68.0% |
| 🟨 JavaScript | 24.4% |
| 🎨 CSS | 7.3% |

---

## 📄 License

This project was built for a hackathon. See repository for license details.
