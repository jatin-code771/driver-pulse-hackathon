import { useState, useEffect } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer
} from "recharts";

function DynamicVelocityChart({ trips }) {

  const [visibleData, setVisibleData] = useState([]);
  const [index, setIndex] = useState(0);

  // prepare dataset
  const timeline = trips
    .map(row => ({
      timestamp: new Date(row.timestamp).toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit"
      }),
      current_velocity: Number(row.current_velocity),
      target_velocity: Number(row.target_velocity),
      status: row.forecast_status
    }))
    .sort((a, b) => a.timestamp.localeCompare(b.timestamp));

  // reset when driver changes
  useEffect(() => {
    setVisibleData([]);
    setIndex(0);
  }, [trips]);

  // animate plotting
  useEffect(() => {

    if (index >= timeline.length) return;

    const interval = setTimeout(() => {

      setVisibleData(prev => [...prev, timeline[index]]);
      setIndex(index + 1);

    }, 800);

    return () => clearTimeout(interval);

  }, [index, timeline]);

  return (

    <ResponsiveContainer width="100%" height={350}>

      <LineChart data={visibleData}>

        <CartesianGrid strokeDasharray="3 3" />

        <XAxis dataKey="timestamp" />

        <YAxis />

        <Tooltip />

        <Line
          type="monotone"
          dataKey="current_velocity"
          stroke="#2563EB"
          strokeWidth={3}
          name="Current Velocity"
        />

        <Line
          type="monotone"
          dataKey="target_velocity"
          stroke="#22C55E"
          strokeWidth={3}
          name="Target Velocity"
        />

      </LineChart>

    </ResponsiveContainer>

  );

}

export default DynamicVelocityChart;