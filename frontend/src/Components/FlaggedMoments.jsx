import { useState, useEffect } from "react";
import Papa from "papaparse";
import { FaMapMarkerAlt, FaExclamationTriangle, FaRoute } from "react-icons/fa";

function findNearestGPS(accelData, tripId, elapsedSec, timestamp) {

  const tripReadings = accelData.filter(r => r.trip_id === tripId);

  if(tripReadings.length === 0) return null;

  const exact = tripReadings.find(r => r.timestamp === timestamp);

  if(exact) return {lat:exact.gps_lat, lon:exact.gps_lon};

  const elapsed = parseFloat(elapsedSec);

  let closest = tripReadings[0];
  let minDiff = Math.abs(parseFloat(closest.elapsed_seconds) - elapsed);

  for(const r of tripReadings){

    const diff = Math.abs(parseFloat(r.elapsed_seconds) - elapsed);

    if(diff < minDiff){
      minDiff = diff;
      closest = r;
    }

  }

  return {lat:closest.gps_lat, lon:closest.gps_lon};

}

function FlaggedMoments({ driverId }) {

  const [data,setData] = useState([]);
  const [filteredData,setFilteredData] = useState([]);
  const [accelData,setAccelData] = useState([]);
  const [tripsData,setTripsData] = useState([]);
  const [viewMode,setViewMode] = useState("cards");

  useEffect(()=>{

    Promise.all([
      fetch("/flagged_moments_latest.csv").then(r=>r.text()),
      fetch("/accelerometer_data.csv").then(r=>r.text()),
      fetch("/trips.csv").then(r=>r.text())
    ]).then(([flags,accel,trips])=>{

      Papa.parse(flags,{
        header:true,
        skipEmptyLines:true,
        complete:(res)=>setData(res.data)
      });

      Papa.parse(accel,{
        header:true,
        skipEmptyLines:true,
        complete:(res)=>setAccelData(res.data)
      });

      Papa.parse(trips,{
        header:true,
        skipEmptyLines:true,
        complete:(res)=>setTripsData(res.data)
      });

    });

  },[]);


  useEffect(()=>{

    if(!driverId) return;

    const tripLocationMap = {};

    tripsData.forEach(t=>{
      tripLocationMap[t.trip_id] = {
        pickup:t.pickup_location,
        dropoff:t.dropoff_location
      };
    });

    const filtered = data
      .filter(r => r.driver_id === driverId)
      .map(row=>{

        const gps = findNearestGPS(
          accelData,
          row.trip_id,
          row.elapsed_seconds,
          row.timestamp
        );

        const tripLoc = tripLocationMap[row.trip_id];

        return {
          ...row,
          gps_lat:gps?.lat || null,
          gps_lon:gps?.lon || null,
          pickup_location:tripLoc?.pickup || "—",
          dropoff_location:tripLoc?.dropoff || "—"
        };

      });

    setFilteredData(filtered);

  },[driverId,data,accelData,tripsData]);


  return (

    <div style={{marginTop:"40px"}}>

      {/* VIEW TOGGLE */}

      <div className="flex justify-end mb-4 gap-2">

        <button
          onClick={()=>setViewMode("cards")}
          className={`px-4 py-2 rounded-lg ${
            viewMode==="cards" ? "bg-blue-500 text-white":"bg-gray-200"
          }`}
        >
          📋 Card View
        </button>

        <button
          onClick={()=>setViewMode("table")}
          className={`px-4 py-2 rounded-lg ${
            viewMode==="table" ? "bg-blue-500 text-white":"bg-gray-200"
          }`}
        >
          📊 Table View
        </button>

      </div>


      {/* ========================= */}
      {/* CARD VIEW */}
      {/* ========================= */}

      {viewMode==="cards" && (

        <div className="space-y-4">

          <h2 className="text-2xl font-semibold mb-4">
            ⚠️ Flagged Moments — Where They Happened
          </h2>

          {filteredData.map((row,index)=>(

            <div
              key={index}
              className={`rounded-xl p-5 border-l-4 shadow-lg ${
                row.severity==="high"
                  ?"border-l-red-500"
                  :row.severity==="medium"
                  ?"border-l-yellow-500"
                  :"border-l-green-500"
              }`}
            >

              <div className="flag-card-layout">

                {/* LEFT INFO */}

                <div>

                  <div className="flex items-center gap-3 mb-2">

                    <FaExclamationTriangle/>

                    <span className="font-bold text-lg">
                      {row.flag_type?.replace(/_/g," ").replace(/\b\w/g,l=>l.toUpperCase())}
                    </span>

                    <span className="px-3 py-1 bg-gray-700 text-white rounded-full text-xs">
                      {row.severity}
                    </span>

                  </div>

                  <p>{row.explanation}</p>

                  <p className="flag-time">
🕒 {new Date(row.timestamp).toLocaleTimeString()}
</p>

                  <p className="text-sm">
                    🚗 {row.trip_id} • {row.flag_id}
                  </p>

                </div>


                {/* RIGHT LOCATION CARD */}

                <div className="bg-gray-700 p-4 rounded-lg">

                  <div className="flex items-center gap-2 mb-2">
                    <FaMapMarkerAlt className="text-red-400"/>
                    <b className="text-white">Flag Location</b>
                  </div>

                  {row.gps_lat && row.gps_lon ? (

                    <>
                      <p className="gps-text">
                        📍 GPS: {parseFloat(row.gps_lat).toFixed(4)}°N, {parseFloat(row.gps_lon).toFixed(4)}°E
                      </p>

                      <a
                        href={`https://www.google.com/maps?q=${row.gps_lat},${row.gps_lon}`}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="map-button"
                      >
                        🗺️ View on Map
                      </a>
                    </>

                  ) : (

                    <p className="text-gray-400">
                      GPS unavailable
                    </p>

                  )}

                  <div className="mt-3">

                    <div className="flex items-center gap-2 text-white">
                      <FaRoute/>
                      <span>Trip Route</span>
                    </div>

                    <p className="text-white">
                      {row.pickup_location} → {row.dropoff_location}
                    </p>

                  </div>

                </div>

              </div>

            </div>

          ))}

        </div>

      )}


      {/* ========================= */}
      {/* TABLE VIEW */}
      {/* ========================= */}

      {viewMode==="table" && (

        <div className="flag-table-container">

          <table className="flag-table">

            <thead>
              <tr>
                <th>Flag ID</th>
                <th>Trip</th>
                <th>Time</th>
                <th>Type</th>
                <th>Severity</th>
                <th>Location</th>
                <th>Route</th>
              </tr>
            </thead>

            <tbody>

              {filteredData.map((row,index)=>(
                
                <tr key={index}>

                  <td>{row.flag_id}</td>

                  <td>{row.trip_id}</td>

                  <td>{row.timestamp}</td>

                  <td>
                    {row.flag_type?.replace(/_/g," ").replace(/\b\w/g,l=>l.toUpperCase())}
                  </td>

                  <td>
                    <span className={`severity-${row.severity}`}>
                      {row.severity}
                    </span>
                  </td>

                  <td>
                    {row.gps_lat && row.gps_lon
                      ? `${parseFloat(row.gps_lat).toFixed(4)}, ${parseFloat(row.gps_lon).toFixed(4)}`
                      : "—"}
                  </td>

                  <td>
                    {row.pickup_location} → {row.dropoff_location}
                  </td>

                </tr>

              ))}

            </tbody>

          </table>

        </div>

      )}

    </div>

  );

}

export default FlaggedMoments;