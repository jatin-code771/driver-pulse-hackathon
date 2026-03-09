import { useState, useEffect } from "react";
import Papa from "papaparse";
import {
  PieChart,
  Pie,
  Cell,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  AreaChart,
  Area,
} from "recharts";
import { FaSun, FaMoon, FaMapMarkerAlt, FaExclamationTriangle, FaRoute, FaInfoCircle, FaShieldAlt } from "react-icons/fa";
import "./index.css";

// Find the nearest accelerometer GPS for a flagged moment.
// Strategy: match by trip_id + exact timestamp first, then fallback to closest elapsed_seconds.
// Uses sensor_id ordering to prefer primary readings when ties occur.
function findNearestGPS(accelData, tripId, elapsedSec, timestamp) {
  const tripReadings = accelData.filter((r) => r.trip_id === tripId);
  if (tripReadings.length === 0) return null;

  // 1) Exact timestamp match (most reliable)
  const exactMatch = tripReadings.find((r) => r.timestamp === timestamp);
  if (exactMatch) return { lat: exactMatch.gps_lat, lon: exactMatch.gps_lon };

  // 2) Closest elapsed_seconds, preferring lower sensor_id on ties (primary sensor)
  const elapsed = parseFloat(elapsedSec);
  let closest = tripReadings[0];
  let minDiff = Math.abs(parseFloat(closest.elapsed_seconds) - elapsed);
  for (const r of tripReadings) {
    const diff = Math.abs(parseFloat(r.elapsed_seconds) - elapsed);
    if (diff < minDiff || (diff === minDiff && r.sensor_id < closest.sensor_id)) {
      minDiff = diff;
      closest = r;
    }
  }
  return { lat: closest.gps_lat, lon: closest.gps_lon };
}

function App() {
  const [driverId, setDriverId] = useState("");
  const [data, setData] = useState([]);
  const [filteredData, setFilteredData] = useState([]);
  const [isDarkMode, setIsDarkMode] = useState(true);
  const [accelData, setAccelData] = useState([]);
  const [tripsData, setTripsData] = useState([]);
  const [viewMode, setViewMode] = useState("cards"); // "cards" or "table"

  useEffect(() => {
    // Load all three datasets from API in parallel
    Promise.all([
      fetch("/api/flagged-moments").then((r) => r.json()),
      fetch("/api/accelerometer").then((r) => r.json()),
      fetch("/api/trips").then((r) => r.json()),
    ]).then(([flagsJson, accelJson, tripsJson]) => {
      setData(flagsJson.filter((r) => r.flag_id));
      setAccelData(accelJson.filter((r) => r.sensor_id));
      setTripsData(tripsJson.filter((r) => r.trip_id));
    });
  }, []);

  // Build a lookup: trip_id -> { pickup_location, dropoff_location }
  const tripLocationMap = {};
  tripsData.forEach((t) => {
    tripLocationMap[t.trip_id] = {
      pickup: t.pickup_location,
      dropoff: t.dropoff_location,
    };
  });

  const handleDriverIdChange = (e) => {
    const id = e.target.value;
    setDriverId(id);

    if (id) {
      const filtered = data
        .filter((row) => row.driver_id === id)
        .map((row) => {
          const gps = findNearestGPS(accelData, row.trip_id, row.elapsed_seconds, row.timestamp);
          const tripLoc = tripLocationMap[row.trip_id];
          return {
            ...row,
            gps_lat: gps?.lat || null,
            gps_lon: gps?.lon || null,
            pickup_location: tripLoc?.pickup || "—",
            dropoff_location: tripLoc?.dropoff || "—",
          };
        });
      setFilteredData(filtered);
    } else {
      setFilteredData([]);
    }
  };

  const severityCounts = filteredData.reduce((acc, row) => {
    if (row && row.severity) {
      acc[row.severity] = (acc[row.severity] || 0) + 1;
    }
    return acc;
  }, {});

  const severityData = Object.entries(severityCounts).map(
    ([severity, count]) => ({
      name: severity.charAt(0).toUpperCase() + severity.slice(1),
      value: count,
    })
  );

  const flagTypeCounts = filteredData.reduce((acc, row) => {
    if (row && row.flag_type) {
      acc[row.flag_type] = (acc[row.flag_type] || 0) + 1;
    }
    return acc;
  }, {});

  const flagTypeData = Object.entries(flagTypeCounts).map(([type, count]) => ({
    type: type ? type.replace(/_/g, " ").replace(/\b\w/g, (l) => l.toUpperCase()) : "Unknown",
    count,
  }));

  // Calculate stats
  const totalFlags = filteredData.length;
  const highSeverity = severityCounts.high || 0;
  const uniqueTrips = new Set(filteredData.map(row => row.trip_id).filter(Boolean)).size;

  const COLORS = ["#3B82F6", "#22C55E", "#FACC15", "#EF4444", "#A855F7"];
  const SEVERITY_COLORS = { High: "#EF4444", Medium: "#F59E0B", Low: "#22C55E" };
  const FLAG_TYPE_COLORS = {
    "Conflict Moment": "#EF4444",
    "Harsh Braking": "#F97316",
    "Audio Spike": "#A855F7",
    "Sustained Stress": "#3B82F6",
    "Moderate Brake": "#22C55E",
  };

  // -------- Driver Risk Score Logic --------

  const severityWeights = {
    low: 1,
    medium: 3,
    high: 5,
  };

  let totalPoints = 0;
  let riskScore = 0;
  let riskLevel = "Low";
  let mainIssue = "None";

  if (filteredData.length > 0) {
    filteredData.forEach(row => {
      if (row && row.severity) {
        totalPoints += severityWeights[row.severity] || 0;
      }
    });

    const maxPossiblePoints = filteredData.length * 5;
    riskScore = maxPossiblePoints ? Math.round((totalPoints / maxPossiblePoints) * 100) : 0;

    if (riskScore > 70) riskLevel = "High";
    else if (riskScore > 40) riskLevel = "Medium";

    // -------- Most Common Issue --------
    if (flagTypeData.length > 0) {
      mainIssue = flagTypeData.reduce((prev, curr) =>
        prev.count > curr.count ? prev : curr
      ).type;
    }
  }
  // Score radar data
  const avgMotion = filteredData.length ? (filteredData.reduce((s, r) => s + parseFloat(r.motion_score || 0), 0) / filteredData.length * 100).toFixed(0) : 0;
  const avgAudio = filteredData.length ? (filteredData.reduce((s, r) => s + parseFloat(r.audio_score || 0), 0) / filteredData.length * 100).toFixed(0) : 0;
  const avgCombined = filteredData.length ? (filteredData.reduce((s, r) => s + parseFloat(r.combined_score || 0), 0) / filteredData.length * 100).toFixed(0) : 0;
  const radarData = [
    { metric: "Motion", value: Number(avgMotion), fullMark: 100 },
    { metric: "Audio", value: Number(avgAudio), fullMark: 100 },
    { metric: "Combined", value: Number(avgCombined), fullMark: 100 },
    { metric: "Risk", value: riskScore, fullMark: 100 },
    { metric: "High %", value: totalFlags ? Math.round((highSeverity / totalFlags) * 100) : 0, fullMark: 100 },
  ];

  // Timeline data (flags sorted by elapsed_seconds)
  const timelineData = [...filteredData]
    .sort((a, b) => parseInt(a.elapsed_seconds) - parseInt(b.elapsed_seconds))
    .map((r, i) => ({
      index: i + 1,
      time: `${Math.floor(parseInt(r.elapsed_seconds) / 60)}m`,
      motion: parseFloat(r.motion_score || 0),
      audio: parseFloat(r.audio_score || 0),
      combined: parseFloat(r.combined_score || 0),
      flag_type: r.flag_type,
    }));

  const toggleTheme = () => {
    setIsDarkMode(!isDarkMode);
  };

  return (
  <div className={`min-h-screen p-10 transition-colors duration-300 ${
    isDarkMode 
      ? "bg-linear-to-br from-gray-900 via-gray-800 to-black text-white" 
      : "bg-linear-to-br from-blue-50 via-indigo-50 to-purple-100 text-gray-900"
  }`}>

    {/* THEME TOGGLE BUTTON */}
    <div className="fixed top-4 right-4 z-50">
      <button
        onClick={toggleTheme}
        className={`p-3 rounded-full shadow-lg transition-all duration-300 transform hover:scale-110 ${
          isDarkMode 
            ? "bg-gray-800 text-yellow-400 hover:bg-gray-700 border border-gray-600" 
            : "bg-white text-gray-800 hover:bg-gray-50 border border-gray-200"
        }`}
      >
        {isDarkMode ? <FaSun size={20} /> : <FaMoon size={20} />}
      </button>
    </div>

    {/* TICKER */}
    <div className={`py-2 mb-6 rounded-lg overflow-hidden border transition-colors duration-300 ${
      isDarkMode 
        ? "bg-gray-800 text-gray-200 border-gray-700" 
        : "bg-indigo-600 text-white border-indigo-500"
    }`}>
      <div className="animate-marquee whitespace-nowrap font-semibold">
        🚗 Driver Safety Analytics Dashboard • Monitor Risk Patterns • Improve Driving Behaviour • Real-Time Safety Insights
      </div>
    </div>

    <div className="max-w-7xl mx-auto">

      {/* HEADER */}
      <h1 className="text-5xl font-extrabold text-center mb-12 bg-linear-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
        Driver Pulse Dashboard
      </h1>

      {/* DRIVER INPUT */}
      <div className={`backdrop-blur-lg shadow-xl rounded-xl p-6 mb-10 border transition hover:shadow-2xl hover:border-blue-400 hover:shadow-blue-500/25 ${
        isDarkMode 
          ? "bg-gray-800/70 border-gray-700" 
          : "bg-white/70 border-gray-200"
      }`}>
        <label className={`block text-lg font-semibold mb-2 ${
          isDarkMode ? "text-gray-200" : "text-gray-700"
        }`}>
          Enter Driver ID
        </label>

        <input
          type="text"
          placeholder="Example: DRV111"
          value={driverId}
          onChange={handleDriverIdChange}
          className={`w-full p-3 border rounded-lg outline-none transition placeholder-gray-400 text-lg ${
            isDarkMode 
              ? "border-gray-600 bg-gray-700 text-white focus:ring-2 focus:ring-blue-400" 
              : "border-gray-300 bg-white text-gray-900 focus:ring-2 focus:ring-indigo-400"
          }`}
        />
        {filteredData.length > 0 && (
          <p className={`mt-2 text-sm ${isDarkMode ? "text-gray-400" : "text-gray-500"}`}>
            Found <span className="font-bold text-blue-400">{filteredData.length}</span> flagged moments across{" "}
            <span className="font-bold text-green-400">{[...new Set(filteredData.map((d) => d.trip_id))].length}</span> trips
          </p>
        )}
      </div>

      {filteredData.length > 0 ? (
        <>
          {/* STATS */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-10">

            <div className={`shadow-lg rounded-xl p-6 border transform transition duration-300 hover:-translate-y-2 hover:shadow-2xl hover:border-blue-400 hover:shadow-blue-500/25 ${
              isDarkMode ? "bg-gray-800 border-gray-700" : "bg-white border-gray-200"
            }`}>
              <h3 className={isDarkMode ? "text-gray-400" : "text-gray-500"}>Total Flags</h3>
              <p className="text-3xl font-bold text-blue-400">{totalFlags}</p>
            </div>

            <div className={`shadow-lg rounded-xl p-6 border transform transition duration-300 hover:-translate-y-2 hover:shadow-2xl hover:border-blue-400 hover:shadow-blue-500/25 ${
              isDarkMode ? "bg-gray-800 border-gray-700" : "bg-white border-gray-200"
            }`}>
              <h3 className={isDarkMode ? "text-gray-400" : "text-gray-500"}>High Severity</h3>
              <p className="text-3xl font-bold text-red-400">{highSeverity}</p>
            </div>

            <div className={`shadow-lg rounded-xl p-6 border transform transition duration-300 hover:-translate-y-2 hover:shadow-2xl hover:border-blue-400 hover:shadow-blue-500/25 ${
              isDarkMode ? "bg-gray-800 border-gray-700" : "bg-white border-gray-200"
            }`}>
              <h3 className={isDarkMode ? "text-gray-400" : "text-gray-500"}>Trips Flagged</h3>
              <p className="text-3xl font-bold text-green-400">
                {[...new Set(filteredData.map((d) => d.trip_id))].length}
              </p>
            </div>

          </div>

          {/* SEVERITY GUIDE */}
          <div className={`shadow-xl rounded-xl p-6 mb-10 border transition hover:shadow-2xl hover:border-blue-400 hover:shadow-blue-500/25 ${
            isDarkMode ? "bg-gray-800 border-gray-700" : "bg-white border-gray-200"
          }`}>
            <div className="flex items-center gap-3 mb-4">
              <FaShieldAlt className="text-blue-400" size={22} />
              <h2 className={`text-xl font-semibold ${isDarkMode ? "text-gray-200" : "text-gray-800"}`}>
                What Does Severity Mean?
              </h2>
            </div>
            <p className={`text-sm mb-4 ${isDarkMode ? "text-gray-400" : "text-gray-500"}`}>
              Each flagged moment is rated by severity based on its <strong>combined score</strong> (a blend of motion + audio sensor data). Here's what each level means for you:
            </p>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {/* HIGH */}
              <div className={`rounded-xl p-4 border-l-4 border-l-red-500 ${isDarkMode ? "bg-red-950/40" : "bg-red-50"}`}>
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-2xl">🔴</span>
                  <span className="font-bold text-red-400 text-lg">HIGH</span>
                  <span className={`text-xs px-2 py-0.5 rounded-full ${isDarkMode ? "bg-red-900 text-red-200" : "bg-red-100 text-red-700"}`}>Score ≥ 0.70</span>
                </div>
                <p className={`text-sm ${isDarkMode ? "text-gray-300" : "text-gray-700"}`}>
                  <strong>Dangerous event.</strong> Harsh braking combined with loud arguments, extreme acceleration, or severe road rage. Immediate attention needed.
                </p>
                <p className={`text-xs mt-2 italic ${isDarkMode ? "text-gray-500" : "text-gray-400"}`}>
                  Example: Sudden hard brake (3.5 m/s²) + passenger argument (95 dB)
                </p>
              </div>
              {/* MEDIUM */}
              <div className={`rounded-xl p-4 border-l-4 border-l-yellow-500 ${isDarkMode ? "bg-yellow-950/40" : "bg-yellow-50"}`}>
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-2xl">🟡</span>
                  <span className="font-bold text-yellow-400 text-lg">MEDIUM</span>
                  <span className={`text-xs px-2 py-0.5 rounded-full ${isDarkMode ? "bg-yellow-900 text-yellow-200" : "bg-yellow-100 text-yellow-700"}`}>Score 0.55 – 0.69</span>
                </div>
                <p className={`text-sm ${isDarkMode ? "text-gray-300" : "text-gray-700"}`}>
                  <strong>Moderate concern.</strong> Notable motion irregularity or elevated cabin noise. Worth reviewing but not critical.
                </p>
                <p className={`text-xs mt-2 italic ${isDarkMode ? "text-gray-500" : "text-gray-400"}`}>
                  Example: Moderate acceleration + slightly elevated audio in cabin
                </p>
              </div>
              {/* LOW */}
              <div className={`rounded-xl p-4 border-l-4 border-l-green-500 ${isDarkMode ? "bg-green-950/40" : "bg-green-50"}`}>
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-2xl">🟢</span>
                  <span className="font-bold text-green-400 text-lg">LOW</span>
                  <span className={`text-xs px-2 py-0.5 rounded-full ${isDarkMode ? "bg-green-900 text-green-200" : "bg-green-100 text-green-700"}`}>Score &lt; 0.55</span>
                </div>
                <p className={`text-sm ${isDarkMode ? "text-gray-300" : "text-gray-700"}`}>
                  <strong>Minor event.</strong> Small irregularity like a quick stop or brief audio spike. Normal driving conditions — just logged for patterns.
                </p>
                <p className={`text-xs mt-2 italic ${isDarkMode ? "text-gray-500" : "text-gray-400"}`}>
                  Example: Brief brake at traffic light with normal cabin environment
                </p>
              </div>
            </div>
            <div className={`mt-4 p-3 rounded-lg flex items-start gap-2 ${isDarkMode ? "bg-blue-950/40 border border-blue-800" : "bg-blue-50 border border-blue-200"}`}>
              <FaInfoCircle className="text-blue-400 mt-0.5 shrink-0" size={16} />
              <p className={`text-xs ${isDarkMode ? "text-gray-400" : "text-gray-500"}`}>
                <strong>How is the score calculated?</strong> Motion score measures driving erraticness from accelerometer data (braking, acceleration, swerving). Audio score measures cabin disturbance from microphone data (shouting, loud music, arguments). The combined score is a weighted blend — higher = more serious.
              </p>
            </div>
          </div>

          {/* Driver Safety Insights */}

          <div className={`shadow-xl rounded-xl p-6 mb-10 border transition hover:shadow-2xl hover:border-blue-400 hover:shadow-blue-500/25 ${
            isDarkMode ? "bg-gray-800 border-gray-700" : "bg-white border-gray-200"
          }`}>

            <h2 className={`text-xl font-semibold mb-6 ${
              isDarkMode ? "text-gray-200" : "text-gray-800"
            }`}>
              Driver Safety Insights
            </h2>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">

              <div>
                <p className={isDarkMode ? "text-gray-400" : "text-gray-500"}>Risk Score</p>

                <p className="text-3xl font-bold text-blue-400 animate-pulse">
                  {riskScore} / 100
                </p>

                <div className={`mt-3 w-full rounded-full h-3 ${
                  isDarkMode ? "bg-gray-700" : "bg-gray-200"
                }`}>
                  <div
                    className="bg-blue-500 h-3 rounded-full transition-all duration-700"
                    style={{ width: `${riskScore}%` }}
                  ></div>
                </div>
              </div>

              <div>
                <p className={isDarkMode ? "text-gray-400" : "text-gray-500"}>Risk Level</p>
                <p
                  className={`text-2xl font-bold ${
                    riskLevel === "High"
                      ? "text-red-400"
                      : riskLevel === "Medium"
                      ? "text-yellow-400"
                      : "text-green-400"
                  }`}
                >
                  {riskLevel}
                </p>
              </div>

              <div>
                <p className={isDarkMode ? "text-gray-400" : "text-gray-500"}>Main Risk Factor</p>
                <p className={`text-lg font-semibold ${
                  isDarkMode ? "text-gray-200" : "text-gray-800"
                }`}>
                  {mainIssue}
                </p>
              </div>

            </div>

          </div>

          {/* VIEW TOGGLE */}
          <div className="flex justify-end mb-4 gap-2">
            <button
              onClick={() => setViewMode("cards")}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition ${
                viewMode === "cards"
                  ? "bg-blue-500 text-white"
                  : isDarkMode ? "bg-gray-700 text-gray-300 hover:bg-gray-600" : "bg-gray-200 text-gray-700 hover:bg-gray-300"
              }`}
            >
              📋 Card View
            </button>
            <button
              onClick={() => setViewMode("table")}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition ${
                viewMode === "table"
                  ? "bg-blue-500 text-white"
                  : isDarkMode ? "bg-gray-700 text-gray-300 hover:bg-gray-600" : "bg-gray-200 text-gray-700 hover:bg-gray-300"
              }`}
            >
              📊 Table View
            </button>
          </div>

          {/* FLAGGED MOMENTS - CARD VIEW */}
          {viewMode === "cards" && (
            <div className="space-y-4 mb-12">
              <h2 className={`text-2xl font-semibold mb-4 ${
                isDarkMode ? "text-gray-200" : "text-gray-800"
              }`}>
                ⚠️ Flagged Moments — Where They Happened
              </h2>
              {filteredData.map((row, index) => (
                <div
                  key={index}
                  className={`rounded-xl p-5 border-l-4 shadow-lg transition hover:shadow-2xl ${
                    row.severity === "high"
                      ? `border-l-red-500 ${isDarkMode ? "bg-gray-800" : "bg-red-50"}`
                      : row.severity === "medium"
                      ? `border-l-yellow-500 ${isDarkMode ? "bg-gray-800" : "bg-yellow-50"}`
                      : `border-l-green-500 ${isDarkMode ? "bg-gray-800" : "bg-green-50"}`
                  }`}
                >
                  <div className="flex flex-wrap items-start justify-between gap-4">
                    {/* Left: Flag info */}
                    <div className="flex-1 min-w-[280px]">
                      <div className="flex items-center gap-3 mb-2">
                        <FaExclamationTriangle className={
                          row.severity === "high" ? "text-red-400" :
                          row.severity === "medium" ? "text-yellow-400" : "text-green-400"
                        } size={18} />
                        <span className={`text-lg font-bold ${isDarkMode ? "text-gray-100" : "text-gray-900"}`}>
                          {row.flag_type?.replace(/_/g, " ").replace(/\b\w/g, (l) => l.toUpperCase())}
                        </span>
                        <span
                          className={`px-3 py-1 rounded-full text-xs font-semibold ${
                            row.severity === "high"
                              ? (isDarkMode ? "bg-red-900 text-red-200" : "bg-red-100 text-red-800")
                              : row.severity === "medium"
                              ? (isDarkMode ? "bg-yellow-900 text-yellow-200" : "bg-yellow-100 text-yellow-800")
                              : (isDarkMode ? "bg-green-900 text-green-200" : "bg-green-100 text-green-800")
                          }`}
                        >
                          {row.severity?.toUpperCase()}
                        </span>
                      </div>
                      <p className={`text-base mb-2 ${isDarkMode ? "text-gray-300" : "text-gray-700"}`}>
                        {row.explanation}
                      </p>
                      <div className={`text-sm space-y-1 ${isDarkMode ? "text-gray-400" : "text-gray-500"}`}>
                        <p>🕐 <strong>Time:</strong> {row.timestamp}</p>
                        <p>🚗 <strong>Trip:</strong> {row.trip_id} &nbsp;•&nbsp; <strong>Flag:</strong> {row.flag_id}</p>
                        <p>📊 <strong>Scores:</strong> Motion {row.motion_score} | Audio {row.audio_score} | Combined {row.combined_score}</p>
                      </div>
                    </div>

                    {/* Right: Location info */}
                    <div className={`min-w-[240px] rounded-lg p-4 ${isDarkMode ? "bg-gray-700/60" : "bg-white"}`}>
                      <div className="flex items-center gap-2 mb-2">
                        <FaMapMarkerAlt className="text-red-400" size={16} />
                        <span className={`font-semibold text-sm ${isDarkMode ? "text-gray-200" : "text-gray-800"}`}>
                          Flag Location
                        </span>
                      </div>
                      {row.gps_lat && row.gps_lon ? (
                        <>
                          <p className={`text-sm mb-1 ${isDarkMode ? "text-gray-300" : "text-gray-700"}`}>
                            📍 <strong>GPS:</strong> {parseFloat(row.gps_lat).toFixed(4)}°N, {parseFloat(row.gps_lon).toFixed(4)}°E
                          </p>
                          <a
                            href={`https://www.google.com/maps?q=${row.gps_lat},${row.gps_lon}`}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="inline-block mt-1 px-3 py-1.5 bg-blue-500 hover:bg-blue-600 text-white text-xs font-semibold rounded-lg transition"
                          >
                            🗺️ View on Map
                          </a>
                        </>
                      ) : (
                        <p className={`text-sm ${isDarkMode ? "text-gray-500" : "text-gray-400"}`}>GPS data not available</p>
                      )}
                      <div className={`mt-3 pt-2 border-t ${isDarkMode ? "border-gray-600" : "border-gray-200"}`}>
                        <div className="flex items-center gap-2">
                          <FaRoute className="text-blue-400" size={14} />
                          <span className={`text-xs font-semibold ${isDarkMode ? "text-gray-300" : "text-gray-600"}`}>Trip Route</span>
                        </div>
                        <p className={`text-sm mt-1 ${isDarkMode ? "text-gray-300" : "text-gray-700"}`}>
                          {row.pickup_location} → {row.dropoff_location}
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* FLAGGED MOMENTS - TABLE VIEW */}
          {viewMode === "table" && (
          <div className={`shadow-xl rounded-xl p-6 mb-12 border transition hover:shadow-2xl hover:border-blue-400 hover:shadow-blue-500/25 ${
            isDarkMode ? "bg-gray-800 border-gray-700" : "bg-white border-gray-200"
          }`}>
            <h2 className={`text-2xl font-semibold mb-6 ${
              isDarkMode ? "text-gray-200" : "text-gray-800"
            }`}>
              Flagged Moments
            </h2>

            <div className="overflow-x-auto">
              <table className="w-full text-left border-collapse">

                <thead>
                  <tr className={isDarkMode ? "bg-gray-700 text-gray-200" : "bg-gray-100 text-gray-700"}>
                    <th className="p-3">Flag ID</th>
                    <th className="p-3">Trip ID</th>
                    <th className="p-3">Timestamp</th>
                    <th className="p-3">Flag Type</th>
                    <th className="p-3">Severity</th>
                    <th className="p-3">Location (GPS)</th>
                    <th className="p-3">Route</th>
                    <th className="p-3">Explanation</th>
                  </tr>
                </thead>

                <tbody>
                  {filteredData.map((row, index) => (
                    <tr
                      key={index}
                      className={`border-b transition ${
                        isDarkMode 
                          ? "border-gray-600 odd:bg-gray-800 hover:bg-gray-700" 
                          : "border-gray-200 odd:bg-gray-50 hover:bg-blue-50"
                      }`}
                    >
                      <td className={`p-3 ${isDarkMode ? "text-gray-300" : "text-gray-900"}`}>{row.flag_id}</td>
                      <td className={`p-3 ${isDarkMode ? "text-gray-300" : "text-gray-900"}`}>{row.trip_id}</td>
                      <td className={`p-3 ${isDarkMode ? "text-gray-300" : "text-gray-900"}`}>{row.timestamp}</td>
                      <td className={`p-3 ${isDarkMode ? "text-gray-300" : "text-gray-900"}`}>
                        {row.flag_type?.replace(/_/g, " ").replace(/\b\w/g, (l) => l.toUpperCase())}
                      </td>

                      <td className="p-3">
                        <span
                          className={`px-3 py-1 rounded-full text-sm font-medium ${
                            row.severity === "high"
                              ? (isDarkMode ? "bg-red-900 text-red-200" : "bg-red-100 text-red-800")
                              : row.severity === "medium"
                              ? (isDarkMode ? "bg-yellow-900 text-yellow-200" : "bg-yellow-100 text-yellow-800")
                              : (isDarkMode ? "bg-green-900 text-green-200" : "bg-green-100 text-green-800")
                          }`}
                        >
                          {row.severity}
                        </span>
                      </td>

                      <td className={`p-3 ${isDarkMode ? "text-gray-300" : "text-gray-900"}`}>
                        {row.gps_lat && row.gps_lon ? (
                          <a
                            href={`https://www.google.com/maps?q=${row.gps_lat},${row.gps_lon}`}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="text-blue-400 hover:underline text-sm"
                          >
                            {parseFloat(row.gps_lat).toFixed(4)}, {parseFloat(row.gps_lon).toFixed(4)}
                          </a>
                        ) : "—"}
                      </td>

                      <td className={`p-3 text-sm ${isDarkMode ? "text-gray-300" : "text-gray-900"}`}>
                        {row.pickup_location} → {row.dropoff_location}
                      </td>

                      <td className={`p-3 ${isDarkMode ? "text-gray-400" : "text-gray-600"}`}>{row.explanation}</td>
                    </tr>
                  ))}
                </tbody>

              </table>
            </div>
          </div>
          )}

          {/* CHARTS */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-10">

            <div className={`shadow-xl rounded-xl p-6 border transition hover:shadow-2xl hover:border-blue-400 hover:shadow-blue-500/25 ${
              isDarkMode ? "bg-gray-800 border-gray-700" : "bg-white border-gray-200"
            }`}>
              <h2 className={`text-xl font-semibold mb-6 ${
                isDarkMode ? "text-gray-200" : "text-gray-800"
              }`}>
                Severity Distribution
              </h2>

              <ResponsiveContainer width="100%" height={320}>
                <PieChart>
                  <Pie
                    data={severityData.length > 0 ? severityData : [{ name: "No Data", value: 1 }]}
                    cx="50%"
                    cy="50%"
                    innerRadius={55}
                    outerRadius={100}
                    paddingAngle={3}
                    label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                    dataKey="value"
                    animationBegin={0}
                    animationDuration={1200}
                  >
                    {(severityData.length > 0 ? severityData : [{ name: "No Data", value: 1 }]).map((entry, index) => (
                      <Cell
                        key={index}
                        fill={SEVERITY_COLORS[entry.name] || "#8884d8"}
                        stroke={isDarkMode ? "#1f2937" : "#fff"}
                        strokeWidth={2}
                      />
                    ))}
                  </Pie>
                  <Tooltip
                    formatter={(value, name) => [`${value} flags`, name]}
                    contentStyle={isDarkMode
                      ? { backgroundColor: '#374151', border: '1px solid #4B5563', borderRadius: '8px', color: '#f9fafb' }
                      : { backgroundColor: '#ffffff', border: '1px solid #E5E7EB', borderRadius: '8px', color: '#374151' }
                    }
                  />
                  <Legend
                    formatter={(value) => <span style={{ color: isDarkMode ? '#d1d5db' : '#374151', fontSize: '13px' }}>{value}</span>}
                  />
                </PieChart>
              </ResponsiveContainer>
            </div>

            <div className={`shadow-xl rounded-xl p-6 border transition hover:shadow-2xl hover:border-blue-400 hover:shadow-blue-500/25 ${
              isDarkMode ? "bg-gray-800 border-gray-700" : "bg-white border-gray-200"
            }`}>
              <h2 className={`text-xl font-semibold mb-6 ${
                isDarkMode ? "text-gray-200" : "text-gray-800"
              }`}>
                Flag Type Distribution
              </h2>

              <ResponsiveContainer width="100%" height={320}>
                <BarChart data={flagTypeData.length > 0 ? flagTypeData : [{ type: "No Data", count: 0 }]}>
                  <CartesianGrid strokeDasharray="3 3" stroke={isDarkMode ? "#4B5563" : "#E5E7EB"} />
                  <XAxis dataKey="type" stroke={isDarkMode ? "#9CA3AF" : "#6B7280"} tick={{ fontSize: 11 }} />
                  <YAxis stroke={isDarkMode ? "#9CA3AF" : "#6B7280"} />
                  <Tooltip
                    cursor={{ fill: isDarkMode ? 'rgba(75,85,99,0.3)' : 'rgba(59,130,246,0.1)' }}
                    formatter={(value) => [`${value} flags`]}
                    contentStyle={isDarkMode
                      ? { backgroundColor: '#374151', border: '1px solid #4B5563', borderRadius: '8px', color: '#f9fafb' }
                      : { backgroundColor: '#ffffff', border: '1px solid #E5E7EB', borderRadius: '8px', color: '#374151' }
                    }
                  />
                  <Bar dataKey="count" radius={[8,8,0,0]} animationDuration={1200}>
                    {flagTypeData.map((entry, index) => (
                      <Cell key={index} fill={FLAG_TYPE_COLORS[entry.type] || COLORS[index % COLORS.length]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>

          </div>

          {/* SECOND ROW OF CHARTS */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-10 mt-10">

            {/* Radar Chart: Driver Score Profile */}
            <div className={`shadow-xl rounded-xl p-6 border transition hover:shadow-2xl hover:border-blue-400 hover:shadow-blue-500/25 ${
              isDarkMode ? "bg-gray-800 border-gray-700" : "bg-white border-gray-200"
            }`}>
              <h2 className={`text-xl font-semibold mb-2 ${isDarkMode ? "text-gray-200" : "text-gray-800"}`}>
                🎯 Driver Risk Profile
              </h2>
              <p className={`text-xs mb-4 ${isDarkMode ? "text-gray-500" : "text-gray-400"}`}>
                Shows average scores across all flagged moments — higher = more risky
              </p>
              <ResponsiveContainer width="100%" height={320}>
                <RadarChart data={radarData} cx="50%" cy="50%" outerRadius="70%">
                  <PolarGrid stroke={isDarkMode ? "#4B5563" : "#E5E7EB"} />
                  <PolarAngleAxis dataKey="metric" stroke={isDarkMode ? "#9CA3AF" : "#6B7280"} tick={{ fontSize: 12 }} />
                  <PolarRadiusAxis angle={90} domain={[0, 100]} stroke={isDarkMode ? "#4B5563" : "#D1D5DB"} tick={{ fontSize: 10 }} />
                  <Radar name="Score" dataKey="value" stroke="#3B82F6" fill="#3B82F6" fillOpacity={0.3} animationDuration={1500} />
                  <Tooltip
                    formatter={(value) => [`${value}%`]}
                    contentStyle={isDarkMode
                      ? { backgroundColor: '#374151', border: '1px solid #4B5563', borderRadius: '8px', color: '#f9fafb' }
                      : { backgroundColor: '#ffffff', border: '1px solid #E5E7EB', borderRadius: '8px', color: '#374151' }
                    }
                  />
                </RadarChart>
              </ResponsiveContainer>
            </div>

            {/* Area Chart: Score Timeline */}
            <div className={`shadow-xl rounded-xl p-6 border transition hover:shadow-2xl hover:border-blue-400 hover:shadow-blue-500/25 ${
              isDarkMode ? "bg-gray-800 border-gray-700" : "bg-white border-gray-200"
            }`}>
              <h2 className={`text-xl font-semibold mb-2 ${isDarkMode ? "text-gray-200" : "text-gray-800"}`}>
                📈 Score Timeline
              </h2>
              <p className={`text-xs mb-4 ${isDarkMode ? "text-gray-500" : "text-gray-400"}`}>
                How motion, audio and combined scores change across flagged events
              </p>
              <ResponsiveContainer width="100%" height={320}>
                <AreaChart data={timelineData}>
                  <defs>
                    <linearGradient id="motionGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#F97316" stopOpacity={0.4}/>
                      <stop offset="95%" stopColor="#F97316" stopOpacity={0}/>
                    </linearGradient>
                    <linearGradient id="audioGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#A855F7" stopOpacity={0.4}/>
                      <stop offset="95%" stopColor="#A855F7" stopOpacity={0}/>
                    </linearGradient>
                    <linearGradient id="combinedGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#3B82F6" stopOpacity={0.4}/>
                      <stop offset="95%" stopColor="#3B82F6" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke={isDarkMode ? "#4B5563" : "#E5E7EB"} />
                  <XAxis dataKey="index" stroke={isDarkMode ? "#9CA3AF" : "#6B7280"} tick={{ fontSize: 11 }} label={{ value: "Flag #", position: "insideBottom", offset: -5, fill: isDarkMode ? "#9CA3AF" : "#6B7280" }} />
                  <YAxis stroke={isDarkMode ? "#9CA3AF" : "#6B7280"} domain={[0, 1]} tick={{ fontSize: 11 }} />
                  <Tooltip
                    formatter={(value, name) => [value.toFixed(2), name.charAt(0).toUpperCase() + name.slice(1)]}
                    contentStyle={isDarkMode
                      ? { backgroundColor: '#374151', border: '1px solid #4B5563', borderRadius: '8px', color: '#f9fafb' }
                      : { backgroundColor: '#ffffff', border: '1px solid #E5E7EB', borderRadius: '8px', color: '#374151' }
                    }
                  />
                  <Legend formatter={(value) => <span style={{ color: isDarkMode ? '#d1d5db' : '#374151', fontSize: '12px' }}>{value.charAt(0).toUpperCase() + value.slice(1)}</span>} />
                  <Area type="monotone" dataKey="motion" stroke="#F97316" fill="url(#motionGrad)" strokeWidth={2} animationDuration={1500} />
                  <Area type="monotone" dataKey="audio" stroke="#A855F7" fill="url(#audioGrad)" strokeWidth={2} animationDuration={1500} />
                  <Area type="monotone" dataKey="combined" stroke="#3B82F6" fill="url(#combinedGrad)" strokeWidth={2} animationDuration={1500} />
                </AreaChart>
              </ResponsiveContainer>
            </div>

          </div>
        </>
      ) : (
        driverId && (
          <div className={`p-6 rounded-xl shadow-lg text-center border transition hover:border-blue-400 hover:shadow-blue-500/25 ${
            isDarkMode ? "bg-gray-800 border-gray-700" : "bg-white border-gray-200"
          }`}>
            <p className={`text-lg ${isDarkMode ? "text-gray-300" : "text-gray-600"}`}>
              No flagged moments found for driver{" "}
              <span className="font-semibold text-blue-400">{driverId}</span>
            </p>
          </div>
        )
      )}
    </div>
  </div>
);
}

export default App;