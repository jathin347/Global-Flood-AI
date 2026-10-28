import React from 'react';
import {
  ResponsiveContainer,
  ComposedChart,
  Bar,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ReferenceLine
} from 'recharts';
import { WeatherResponse } from '../types';

interface WeatherChartProps {
  weather: WeatherResponse | null;
}

export const WeatherChart: React.FC<WeatherChartProps> = ({ weather }) => {
  if (!weather) {
    return (
      <div className="h-64 flex items-center justify-center border border-slate-800 rounded-xl bg-slate-950/40 text-slate-500 text-xs">
        No weather data loaded. Select a location to inspect time-series.
      </div>
    );
  }

  // Combine historical observations and forecast points into a single timeline
  const hist = weather.historical_observations_168h.hourly;
  const fc = weather.forecast_48h.hourly;

  // Downsample historical observations (take every 4th hour for readability if 168h)
  const chartData: any[] = [];

  hist.forEach((pt, idx) => {
    if (idx % 3 === 0 || idx >= hist.length - 12) {
      chartData.push({
        time: pt.timestamp.slice(5, 13).replace('T', ' '),
        histPrecip: pt.precipitation_mm ?? 0,
        forecastPrecip: null,
        temp: pt.temperature_2m_c,
        soil: pt.soil_moisture_m3m3 ? pt.soil_moisture_m3m3 * 100 : null,
        isForecast: false
      });
    }
  });

  const splitIndex = chartData.length - 1;

  fc.forEach((pt) => {
    chartData.push({
      time: pt.timestamp.slice(5, 13).replace('T', ' '),
      histPrecip: null,
      forecastPrecip: pt.precipitation_mm ?? 0,
      temp: pt.temperature_2m_c,
      soil: null,
      isForecast: true
    });
  });

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 shadow-xl">
      <div className="flex items-center justify-between mb-3">
        <div>
          <h4 className="text-xs font-bold uppercase tracking-wider text-slate-300">
            Precipitation & Thermal Time-Series
          </h4>
          <p className="text-[11px] text-slate-400">
            Strict separation: Historical Observations (Left) vs 48h Numerical Forecast (Right)
          </p>
        </div>
        <div className="flex items-center gap-3 text-[11px]">
          <div className="flex items-center gap-1.5">
            <span className="w-3 h-3 bg-sky-500/80 rounded-sm"></span>
            <span className="text-slate-300">Historical Rain (mm)</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-3 h-3 bg-indigo-500 rounded-sm"></span>
            <span className="text-indigo-400 font-semibold">Forecast Rain (mm)</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-0.5 bg-amber-400"></span>
            <span className="text-amber-400">Temp (°C)</span>
          </div>
        </div>
      </div>

      <div className="h-56 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
            <XAxis dataKey="time" stroke="#64748b" tick={{ fontSize: 10 }} />
            <YAxis yAxisId="left" stroke="#64748b" tick={{ fontSize: 10 }} label={{ value: 'Rain (mm)', angle: -90, position: 'insideLeft', fill: '#64748b', fontSize: 10 }} />
            <YAxis yAxisId="right" orientation="right" stroke="#f59e0b" tick={{ fontSize: 10 }} domain={[-10, 45]} />
            <Tooltip
              contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '0.75rem', fontSize: '11px' }}
              labelStyle={{ color: '#94a3b8' }}
            />
            {splitIndex >= 0 && (
              <ReferenceLine
                yAxisId="left"
                x={chartData[splitIndex]?.time}
                stroke="#06b6d4"
                strokeDasharray="4 4"
                label={{ value: 'Forecast Horizon Starts', fill: '#06b6d4', fontSize: 10, position: 'top' }}
              />
            )}
            <Bar yAxisId="left" dataKey="histPrecip" fill="#38bdf8" name="Observed Rain (mm)" opacity={0.7} />
            <Bar yAxisId="left" dataKey="forecastPrecip" fill="#6366f1" name="Forecast Rain (mm)" />
            <Line yAxisId="right" type="monotone" dataKey="temp" stroke="#f59e0b" dot={false} strokeWidth={1.5} name="Temperature (°C)" />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};
