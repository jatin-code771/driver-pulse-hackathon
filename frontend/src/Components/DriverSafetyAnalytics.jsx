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
  Area
} from "recharts";

function DriverSafetyAnalytics({ driverId }) {

  const [data, setData] = useState([]);
  const [filteredData, setFilteredData] = useState([]);

  useEffect(() => {

    fetch("/flagged_moments_latest.csv")
      .then(res => res.text())
      .then(csv => {

        Papa.parse(csv,{
          header:true,
          skipEmptyLines:true,
          complete:(results)=>{
            setData(results.data)
          }
        })

      })

  },[])

  useEffect(()=>{

    if(!driverId) return

    const filtered = data.filter(
      r => r.driver_id === driverId
    )

    setFilteredData(filtered)

  },[driverId,data])


  // ------------------------
  // Severity Counts
  // ------------------------

  const severityCounts = filteredData.reduce((acc,row)=>{
    const sev = row.severity || "low"
    acc[sev] = (acc[sev] || 0) + 1
    return acc
  },{})

  const severityData = Object.entries(severityCounts).map(
    ([severity,count])=>({
      name:severity,
      value:count
    })
  )


  // ------------------------
  // Flag Type Distribution
  // ------------------------

  const flagCounts = filteredData.reduce((acc,row)=>{
    const type = row.flag_type || "unknown"
    acc[type] = (acc[type] || 0) + 1
    return acc
  },{})

  const flagTypeData = Object.entries(flagCounts).map(
    ([type,count])=>({
      type,
      count
    })
  )


  // ------------------------
  // Risk Score Calculation
  // ------------------------

  const severityWeights = {
    low:1,
    medium:3,
    high:5
  }

  let totalPoints = 0

  filteredData.forEach(row=>{
    totalPoints += severityWeights[row.severity] || 0
  })

  const maxPoints = filteredData.length * 5

  const riskScore =
    maxPoints === 0 ? 0 : Math.round((totalPoints/maxPoints)*100)

  let riskLevel = "Low"

  if(riskScore > 70) riskLevel = "High"
  else if(riskScore > 40) riskLevel = "Medium"


  const mainIssue =
    flagTypeData.length > 0
      ? flagTypeData.reduce((a,b)=>a.count>b.count?a:b).type
      : "None"


  // ------------------------
  // Radar Chart Data
  // ------------------------

 const avgMotion =
  filteredData.length
    ? (filteredData.reduce((s,r)=>s+parseFloat(r.motion_score||0),0)/filteredData.length)*100
    : 0;

const avgAudio =
  filteredData.length
    ? (filteredData.reduce((s,r)=>s+parseFloat(r.audio_score||0),0)/filteredData.length)*100
    : 0;

const avgCombined =
  filteredData.length
    ? (filteredData.reduce((s,r)=>s+parseFloat(r.combined_score||0),0)/filteredData.length)*100
    : 0;

const highPercent =
  filteredData.length
    ? (filteredData.filter(r=>r.severity==="high").length/filteredData.length)*100
    : 0;

const radarData = [
  { metric:"Motion", value:avgMotion },
  { metric:"Audio", value:avgAudio },
  { metric:"Combined", value:avgCombined },
  { metric:"Risk", value:riskScore },
  { metric:"High %", value:highPercent }
];
  // ------------------------
  // Timeline Data
  // ------------------------

  const timelineData = filteredData.map((r,i)=>({
    index:i+1,
    motion:parseFloat(r.motion_score || 0),
    audio:parseFloat(r.audio_score || 0),
    combined:parseFloat(r.combined_score || 0)
  }))


  const COLORS = ["#22c55e","#f59e0b","#ef4444"]

// colors for each flag type
const FLAG_TYPE_COLORS = {
  moderate_brake: "#22c55e",     // green
  sustained_stress: "#3b82f6",   // blue
  conflict_moment: "#ef4444",    // red
  harsh_braking: "#f97316",      // orange
  audio_spike: "#a855f7"         // purple
}


  return (

    <div style={{marginTop:"40px"}}>

      {/* ------------------------ */}
      {/* Driver Safety Insights */}
      {/* ------------------------ */}

      <div
        style={{
          background:"#1f2937",
          color:"white",
          padding:"25px",
          borderRadius:"10px",
          marginBottom:"30px"
        }}
      >

        <h3 style={{marginBottom:"15px"}}>Driver Safety Insights</h3>

        <div
          style={{
            display:"grid",
            gridTemplateColumns:"1fr 1fr 1fr",
            gap:"20px"
          }}
        >

          <div>

            <p>Risk Score</p>

            <h2 style={{color:"#60a5fa"}}>
              {riskScore} / 100
            </h2>

            <div
              style={{
                height:"10px",
                background:"#374151",
                borderRadius:"10px"
              }}
            >

              <div
                style={{
                  width:`${riskScore}%`,
                  height:"10px",
                  background:"#3b82f6",
                  borderRadius:"10px"
                }}
              />

            </div>

          </div>

          <div>

            <p>Risk Level</p>

            <h2
              style={{
                color:
                  riskLevel==="High"
                  ? "#ef4444"
                  : riskLevel==="Medium"
                  ? "#f59e0b"
                  : "#22c55e"
              }}
            >
              {riskLevel}
            </h2>

          </div>

          <div>

            <p>Main Risk Factor</p>

            <h2 style={{color:"#e5e7eb"}}>
              {mainIssue}
            </h2>

          </div>

        </div>

      </div>


      {/* ------------------------ */}
      {/* Charts Row 1 */}
      {/* ------------------------ */}

      <div
        style={{
          display:"grid",
          gridTemplateColumns:"1fr 1fr",
          gap:"30px"
        }}
      >

        <div>

          <h3>Severity Distribution</h3>

          <ResponsiveContainer width="100%" height={300}>

            <PieChart>

              <Pie
                data={severityData}
                dataKey="value"
                nameKey="name"
                innerRadius={60}
                outerRadius={100}
              >

                {severityData.map((entry,index)=>(
                  <Cell
                    key={index}
                    fill={COLORS[index % COLORS.length]}
                  />
                ))}

              </Pie>

              <Tooltip/>

              <Legend/>

            </PieChart>

          </ResponsiveContainer>

        </div>


        <div>

          <h3>Flag Type Distribution</h3>

          <ResponsiveContainer width="100%" height={300}>

            <BarChart data={flagTypeData}>

  <CartesianGrid strokeDasharray="3 3"/>

  <XAxis dataKey="type"/>

  <YAxis/>

  <Tooltip/>

  <Bar dataKey="count">

    {flagTypeData.map((entry,index)=>(

      <Cell
        key={index}
        fill={
          FLAG_TYPE_COLORS[entry.type] ||
          COLORS[index % COLORS.length]
        }
      />

    ))}

  </Bar>

</BarChart>

          </ResponsiveContainer>

        </div>

      </div>


      {/* ------------------------ */}
      {/* Charts Row 2 */}
      {/* ------------------------ */}

      <div
        style={{
          display:"grid",
          gridTemplateColumns:"1fr 1fr",
          gap:"30px",
          marginTop:"30px"
        }}
      >

        <div>

          <h3>Driver Risk Profile</h3>

          <ResponsiveContainer width="100%" height={320}>

  <RadarChart data={radarData} outerRadius="70%">

    <PolarGrid stroke="#4b5563"/>

    <PolarAngleAxis
      dataKey="metric"
      stroke="#9ca3af"
      tick={{fontSize:12}}
    />

    <PolarRadiusAxis
      angle={90}
      domain={[0,100]}
      stroke="#4b5563"
      tick={{fontSize:10}}
    />

    <Radar
      dataKey="value"
      stroke="#3b82f6"
      fill="#3b82f6"
      fillOpacity={0.35}
      strokeWidth={2}
    />

    <Tooltip/>

  </RadarChart>

</ResponsiveContainer>

        </div>


        <div>

          <h3>Score Timeline</h3>

          <ResponsiveContainer width="100%" height={300}>

            <AreaChart data={timelineData}>

              <CartesianGrid strokeDasharray="3 3"/>

              <XAxis dataKey="index"/>

              <YAxis/>

              <Tooltip/>

              <Area
                type="monotone"
                dataKey="motion"
                stroke="#f97316"
                fill="#f97316"
              />

              <Area
                type="monotone"
                dataKey="audio"
                stroke="#a855f7"
                fill="#a855f7"
              />

              <Area
                type="monotone"
                dataKey="combined"
                stroke="#3b82f6"
                fill="#3b82f6"
              />

            </AreaChart>

          </ResponsiveContainer>

        </div>

      </div>

    </div>

  )
}

export default DriverSafetyAnalytics