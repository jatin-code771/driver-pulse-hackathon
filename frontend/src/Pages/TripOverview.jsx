import { useState, useEffect } from "react";
import Papa from "papaparse";
import DynamicVelocityChart from "../Components/DynamicVelocityChart";
import DriverSafetyAnalytics from "../Components/DriverSafetyAnalytics";
import FlaggedMoments from "../Components/FlaggedMoments";
import { FaShieldAlt, FaInfoCircle } from "react-icons/fa";
import "../styles/TripOverview.css";

function TripOverview() {

  const [isDarkMode, setIsDarkMode] = useState(false);
  const [driverId, setDriverId] = useState("");

  const [tripSummary, setTripSummary] = useState([]);
  const [realtimeData, setRealtimeData] = useState([]);

  const [driverTrips, setDriverTrips] = useState([]);
  const [driverRealtime, setDriverRealtime] = useState([]);

  const [lastUpdated, setLastUpdated] = useState("");

  // =============================
  // LOAD TRIP SUMMARY
  // =============================
  const toggleDarkMode = () => {
  setIsDarkMode(!isDarkMode);
};

  useEffect(() => {

    fetch("/trip_summary.csv")
      .then(res => res.text())
      .then(csv => {

        Papa.parse(csv, {
          header: true,
          skipEmptyLines: true,
          complete: (results) => {

            const cleaned = results.data.filter(
              r => r.driver_id
            );

            setTripSummary(cleaned);

          }
        });

      });

  }, []);

  // =============================
  // LOAD REALTIME DATA
  // =============================

  useEffect(() => {

    fetch("/realtime_driver_predictions.csv")
      .then(res => res.text())
      .then(csv => {

        Papa.parse(csv, {
          header: true,
          skipEmptyLines: true,
          complete: (results) => {

            const cleaned = results.data.filter(
              r => r.driver_id && r.timestamp
            );

            setRealtimeData(cleaned);
setLastUpdated(new Date().toLocaleTimeString());

          }
        });

      });

  }, []);

  // =============================
  // FILTER DRIVER
  // =============================

  const handleDriverChange = (e) => {

    const id = e.target.value.toUpperCase();
    setDriverId(id);

    // summary data
    const trips = tripSummary.filter(
      r => r.driver_id === id
    );

    setDriverTrips(trips);

    // realtime velocity data
    const realtime = realtimeData
      .filter(r => r.driver_id === id)
      .sort(
        (a, b) =>
          new Date(a.timestamp) - new Date(b.timestamp)
      );

    setDriverRealtime(realtime);

  };

  // =============================
  // METRICS
  // =============================

  const totalTrips = driverTrips.length;

  const totalFlags = driverTrips.reduce(
    (sum, r) =>
      sum + Number(r.flagged_moments_count || 0),
    0
  );

  const tripsFlagged = driverTrips.filter(
    r => Number(r.flagged_moments_count) > 0
  ).length;

  const earningVelocity =
    driverTrips.length > 0
      ? driverTrips[0].earnings_velocity
      : "-";

  const severityLevels = {
    none: 0,
    low: 1,
    medium: 2,
    high: 3
  };

  const maxSeverity = driverTrips.reduce((max, trip) => {

    const sev = trip.max_severity || "none";

    return severityLevels[sev] > severityLevels[max]
      ? sev
      : max;

  }, "none");

  // =============================
  // REALTIME STATUS
  // =============================

  const latestRealtime =
    driverRealtime.length > 0
      ? driverRealtime[driverRealtime.length - 1]
      : null;

  const status =
    latestRealtime
      ? latestRealtime.forecast_status
      : "-";


      // EXPORT 
      const exportCSV = () => {

  const rows = [
    ["Trip ID","Flags","Severity","Velocity"]
  ];

  driverTrips.forEach(trip=>{
    rows.push([
      trip.trip_id,
      trip.flagged_moments_count,
      trip.max_severity,
      trip.earnings_velocity
    ]);
  });

  const csvContent =
    "data:text/csv;charset=utf-8," +
    rows.map(e=>e.join(",")).join("\n");

  const link = document.createElement("a");
  link.setAttribute("href",encodeURI(csvContent));
  link.setAttribute("download","driver_report.csv");
  document.body.appendChild(link);
  link.click();
};
  // =============================
  // DRIVER SUMMARY
  // =============================
   const driverName =
  driverRealtime.length > 0
    ? driverRealtime[0].driver_name
    : driverId;

  let summary = "";

// PERFECT DRIVER
if (totalFlags === 0 && status === "ahead") {

  summary =
  `Great work ${driverName}! Your driving behaviour looks excellent and your earnings pace is ahead of the target. Keep maintaining this smooth and safe driving style.`;

}

// SAFE DRIVER
else if (totalFlags <= 2 && maxSeverity !== "high") {

  summary =
  `${driverName}, your driving behaviour is generally safe with only a few minor flagged events. Staying attentive and maintaining smooth braking will help keep your safety score high.`;

}

// MODERATE RISK
else if (totalFlags > 2 && maxSeverity === "medium") {

  summary =
  `${driverName}, we noticed several moderate driving events during recent trips. Reducing sudden braking or stress-related driving patterns could improve passenger comfort and safety.`;

}

// HIGH SEVERITY EVENTS
else if (maxSeverity === "high") {

  summary =
  `${driverName}, some high-severity driving events were detected. Consider maintaining smoother acceleration and braking to improve both safety and ride experience.`;

}

// EARNINGS AT RISK
else if (status === "at_risk" || status === "critical") {

  summary =
  `${driverName}, your current earnings pace is behind the target. Accepting more trips during peak demand hours could help you reach your earnings goal faster.`;

}

// DEFAULT FALLBACK
else {

  summary =
  `${driverName}, your performance is being monitored. Maintaining consistent driving behaviour and trip acceptance will help improve both earnings and safety metrics.`;

}

let aiMessage = "";

if(status === "critical" || status === "at_risk"){
  aiMessage =
  `${driverName}, your earnings pace is currently behind the target. Consider increasing trip frequency during peak hours to stay on track.`;
}

else if(totalFlags >= 3){
  aiMessage =
  `${driverName}, multiple driving events were detected. Maintaining smoother braking and reducing stress events could improve safety and passenger comfort.`;
}

else if(totalFlags > 0){
  aiMessage =
  `${driverName}, your driving is generally safe but a few flagged moments were detected. Staying attentive and maintaining smooth driving will help improve your safety score.`;
}

else{
  aiMessage =
  `Great work ${driverName}! Your driving performance and earnings pace look excellent. Keep maintaining this high standard!`;
}

  // =============================
  // UI
  // =============================

return (

<div className={`app-wrapper ${isDarkMode ? "dark-theme" : "light-theme"}`}>

  {/* HEADER BAR */}

  <div className={`header-bar ${isDarkMode ? "dark" : "light"}`}>

      <div className="ticker-wrapper">
        <div className="ticker-text">
          🚗 Driver Safety Analytics Dashboard • Monitor Risk Patterns • Improve Driving Behaviour • Real-Time Safety Insights
        </div>
      </div>

      <button
        onClick={toggleDarkMode}
        className="dark-toggle"
      >
        {isDarkMode ? "☀ Light Mode" : "🌙 Dark Mode"}
      </button>

    </div>


    <div className="trip-container">

      <h1 className="title">
        Driver Trip Overview
      </h1>
      

      <input
        className="driver-input"
        placeholder="Enter Driver ID (Example: DRV004)"
        value={driverId}
        onChange={handleDriverChange}
      />
      <div className="action-bar">


</div>

      {driverTrips.length > 0 && (

        <>
          <div className="dashboard-actions">

<p className="last-updated">
Last Updated: {lastUpdated}
</p>

<button onClick={exportCSV} className="export-btn">
⬇ Export Report
</button>

</div>
          {/* OVERVIEW CARDS */}

          <div className="overview-grid">

            <div className="overview-card">
              <div className="card-text">
                <p>Total Trips</p>
                <h2>{totalTrips}</h2>
              </div>
              <img src="/icons/trip.png" className="card-icon" />
            </div>

            <div className="overview-card">
              <div className="card-text">
                <p>Total Flags</p>
                <h2>{totalFlags}</h2>
              </div>
              <img src="/icons/flag.png" className="card-icon" />
            </div>

            <div className="overview-card">
              <div className="card-text">
                <p>Trips Flagged</p>
                <h2>{tripsFlagged}</h2>
              </div>
              <img src="/icons/warning.png" className="card-icon" />
            </div>

            <div className="overview-card">
              <div className="card-text">
                <p>Max Severity</p>
                <h2>{maxSeverity}</h2>
              </div>
              <img src="/icons/severity.png" className="card-icon" />
            </div>

            <div className="overview-card">
              <div className="card-text">
                <p>Earnings Velocity</p>
                <h2>{earningVelocity}</h2>
              </div>
              <img src="/icons/velocity.png" className="card-icon" />
            </div>

            <div className="overview-card">
              <div className="card-text">
                <p>Status</p>
                <h2>{status}</h2>
              </div>
              <img src="/icons/status.png" className="card-icon" />
            </div>

          </div>


          <div className="summary-box">
            <h3>Driver Behaviour Insight</h3>
            <p>{summary}</p>
          </div>


          <div className="status-indicator">
            Driver Status:
            <span className={`status-${status.replace("_","-")}`}>
              {status}
            </span>
          </div>


          {/* VELOCITY SECTION */}

          <div className="velocity-section">

            <div className="chart-box">

              <h3>Trip Velocity Progress</h3>

              <p className="chart-subtitle">
                Current velocity vs target velocity over time
              </p>

              <DynamicVelocityChart trips={driverRealtime} />

            </div>


            <div className="velocity-table-box">

              <h3>Driver Velocity Timeline</h3>

              <table className="velocity-table">

                <thead>
                  <tr>
                    <th>Timestamp</th>
                    <th>Earning Velocity</th>
                    <th>Target Velocity</th>
                    <th>Trips Needed</th>
                    <th>Status</th>
                  </tr>
                </thead>

                <tbody>

                  {driverRealtime.map((row,index)=>(

                    <tr key={index}>

                      <td>
                        {new Date(row.timestamp).toLocaleTimeString([],{
                          hour:"2-digit",
                          minute:"2-digit"
                        })}
                      </td>

                      <td>{Number(row.current_velocity).toFixed(2)}</td>

                      <td>{Number(row.target_velocity).toFixed(2)}</td>

                      <td>{row.trips_to_goal}</td>

                      <td>
                        <span className={`status-${row.forecast_status?.replace("_","-")}`}>
                          {row.forecast_status}
                        </span>
                      </td>

                    </tr>

                  ))}

                </tbody>

              </table>

            </div>

          </div>


          <DriverSafetyAnalytics driverId={driverId} />
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

          <FlaggedMoments driverId={driverId} />


          <div className="ai-suggestion-box">
            <h3>🤖 AI Driver Insight</h3>
            <p>{aiMessage}</p>
          </div>

        </>

      )}

      {driverId && driverTrips.length === 0 && (

        <div className="summary-box">
          <p>
            No trip data found for driver <b>{driverId}</b>
          </p>
        </div>

      )}

    </div>

  </div>
);
}

export default TripOverview;