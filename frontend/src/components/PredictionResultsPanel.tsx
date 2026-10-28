import React from 'react';
import {
  CloudRain,
  Satellite,
  Mountain,
  Download,
  Info,
  CheckCircle2,
  AlertCircle,
  TrendingUp,
  TrendingDown,
  Minus
} from 'lucide-react';
import { PredictResponse, WeatherResponse, SatelliteResponse, TerrainResponse, RiskCategory } from '../types';

interface PredictionResultsPanelProps {
  prediction: PredictResponse | null;
  weather: WeatherResponse | null;
  satellite: SatelliteResponse | null;
  terrain: TerrainResponse | null;
  isLoading: boolean;
}

export const PredictionResultsPanel: React.FC<PredictionResultsPanelProps> = ({
  prediction,
  weather,
  satellite,
  terrain,
  isLoading
}) => {
  const getRiskBadge = (category: RiskCategory) => {
    switch (category) {
      case 'low':
        return {
          bg: 'bg-emerald-500/10 border-emerald-500/40 text-emerald-400',
          label: 'LOW FLOOD RISK',
          sub: 'Hydrological conditions within safe baseline'
        };
      case 'moderate':
        return {
          bg: 'bg-amber-500/10 border-amber-500/40 text-amber-400',
          label: 'MODERATE RISK',
          sub: 'Elevated precipitation or localized low slope accumulation'
        };
      case 'high':
        return {
          bg: 'bg-orange-500/10 border-orange-500/40 text-orange-400',
          label: 'HIGH FLOOD RISK',
          sub: 'Intense forecast rainfall combined with saturated soil / lowland'
        };
      case 'very_high':
        return {
          bg: 'bg-rose-500/10 border-rose-500/40 text-rose-400 animate-pulse',
          label: 'VERY HIGH FLOOD RISK',
          sub: 'Severe multi-modal convergence: extreme rainfall & high wetness'
        };
      default:
        return {
          bg: 'bg-slate-800 border-slate-700 text-slate-400',
          label: 'UNAVAILABLE / NOT RUN',
          sub: 'Awaiting coordinate selection or prediction execution'
        };
    }
  };

  const badge = prediction ? getRiskBadge(prediction.risk_category) : getRiskBadge('unavailable');
  const probPercent = prediction?.flood_probability ? (prediction.flood_probability * 100).toFixed(1) : null;

  const downloadJSON = () => {
    if (!prediction) return;
    const dataStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(prediction, null, 2));
    const a = document.createElement('a');
    a.href = dataStr;
    a.download = `GlobalFlood_Prediction_${prediction.location.latitude}_${prediction.location.longitude}.json`;
    a.click();
  };

  const downloadCSV = () => {
    if (!prediction) return;
    const rows = [
      ['Metric', 'Value'],
      ['Latitude', prediction.location.latitude],
      ['Longitude', prediction.location.longitude],
      ['Forecast Horizon (Hours)', prediction.forecast_horizon_hours],
      ['Flood Probability', prediction.flood_probability ?? 'N/A'],
      ['Risk Category', prediction.risk_category],
      ['Model Version', prediction.model_version],
      ['Prediction Timestamp', prediction.prediction_timestamp],
      ['Forecast 48h Rain (mm)', weather?.summary.forecast_48h_total_rain_mm ?? 'N/A'],
      ['Observed 7d Rain (mm)', weather?.summary.past_7d_total_rain_mm ?? 'N/A'],
      ['Elevation (m)', terrain?.elevation_m ?? 'N/A'],
      ['Slope (degrees)', terrain?.slope_degrees ?? 'N/A']
    ];
    const csvContent = 'data:text/csv;charset=utf-8,' + rows.map(e => e.join(',')).join('\n');
    const a = document.createElement('a');
    a.href = encodeURI(csvContent);
    a.download = `GlobalFlood_Prediction_${prediction.location.latitude}_${prediction.location.longitude}.csv`;
    a.click();
  };

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl flex flex-col gap-5 text-slate-200">
      {/* 1. Header with Risk Category Badge */}
      <div>
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
            48-Hour Flood Assessment
          </span>
          {isLoading && (
            <span className="text-[10px] text-cyan-400 font-mono animate-pulse">
              Syncing inputs...
            </span>
          )}
          {prediction?.status === 'demo' && (
            <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-amber-500/20 text-amber-300 border border-amber-500/40">
              SIMULATION DEMO
            </span>
          )}
        </div>

        <div className={`p-4 rounded-xl border ${badge.bg} flex flex-col gap-1`}>
          <div className="flex items-center justify-between">
            <span className="font-extrabold text-lg tracking-wide">{badge.label}</span>
            {probPercent && (
              <span className="font-mono text-2xl font-black">
                {probPercent}%
              </span>
            )}
          </div>
          <p className="text-xs opacity-90">{badge.sub}</p>
        </div>
      </div>

      {/* 2. Probability Gauge Bar & Uncertainty */}
      {prediction?.flood_probability !== null && prediction?.flood_probability !== undefined && (
        <div className="bg-slate-950 p-3.5 rounded-xl border border-slate-800/80">
          <div className="flex justify-between items-center text-xs mb-1.5 font-mono">
            <span className="text-slate-400">Calibrated Probability</span>
            <span className="text-cyan-400 font-bold">{probPercent}%</span>
          </div>
          <div className="w-full bg-slate-800 h-2.5 rounded-full overflow-hidden">
            <div
              className={`h-full transition-all duration-700 ${
                prediction.risk_category === 'very_high'
                  ? 'bg-rose-500'
                  : prediction.risk_category === 'high'
                  ? 'bg-orange-500'
                  : prediction.risk_category === 'moderate'
                  ? 'bg-amber-500'
                  : 'bg-emerald-500'
              }`}
              style={{ width: `${Math.min(100, Math.max(2, (prediction.flood_probability || 0) * 100))}%` }}
            ></div>
          </div>
          {prediction.uncertainty && (
            <div className="mt-2 pt-2 border-t border-slate-800 flex justify-between text-[11px] text-slate-400 font-mono">
              <span>90% CI: {(prediction.uncertainty.lower_bound_90ci * 100).toFixed(0)}% – {(prediction.uncertainty.upper_bound_90ci * 100).toFixed(0)}%</span>
              <span>Confidence: {(prediction.uncertainty.confidence_score * 100).toFixed(0)}%</span>
            </div>
          )}
        </div>
      )}

      {/* 3. Input Modality Freshness & Sources */}
      <div>
        <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2.5">
          Modality Provenance & Status
        </h4>
        <div className="grid grid-cols-1 gap-2">
          {/* Weather */}
          <div className="flex items-center justify-between p-2.5 rounded-lg bg-slate-950 border border-slate-800 text-xs">
            <div className="flex items-center gap-2">
              <CloudRain className="w-4 h-4 text-sky-400" />
              <div>
                <span className="font-medium text-slate-200 block">Open-Meteo NWP Forecast</span>
                <span className="text-[10px] text-slate-400">168h Obs + 48h Forecast</span>
              </div>
            </div>
            <div className="flex items-center gap-1 text-[11px]">
              {weather ? (
                <span className="text-emerald-400 flex items-center gap-1 font-mono">
                  <CheckCircle2 className="w-3.5 h-3.5" /> Valid
                </span>
              ) : (
                <span className="text-slate-500 flex items-center gap-1 font-mono">
                  <AlertCircle className="w-3.5 h-3.5" /> Pending
                </span>
              )}
            </div>
          </div>

          {/* Sentinel-1 Satellite */}
          <div className="flex items-center justify-between p-2.5 rounded-lg bg-slate-950 border border-slate-800 text-xs">
            <div className="flex items-center gap-2">
              <Satellite className="w-4 h-4 text-cyan-400" />
              <div>
                <span className="font-medium text-slate-200 block">Sentinel-1 C-SAR</span>
                <span className="text-[10px] text-slate-400">
                  {satellite?.metadata?.mode || 'IW'} GRD (VV/VH 10m)
                </span>
              </div>
            </div>
            <div className="flex items-center gap-1 text-[11px]">
              {satellite?.status === 'available' ? (
                <span className="text-emerald-400 flex items-center gap-1 font-mono">
                  <CheckCircle2 className="w-3.5 h-3.5" /> Available
                </span>
              ) : satellite?.status === 'coverage_confirmed' ? (
                <span className="text-cyan-400 flex items-center gap-1 font-mono">
                  <CheckCircle2 className="w-3.5 h-3.5" /> Orbit Track
                </span>
              ) : (
                <span className="text-slate-500 flex items-center gap-1 font-mono">
                  <AlertCircle className="w-3.5 h-3.5" /> Check API
                </span>
              )}
            </div>
          </div>

          {/* Topography */}
          <div className="flex items-center justify-between p-2.5 rounded-lg bg-slate-950 border border-slate-800 text-xs">
            <div className="flex items-center gap-2">
              <Mountain className="w-4 h-4 text-amber-400" />
              <div>
                <span className="font-medium text-slate-200 block">Copernicus DEM</span>
                <span className="text-[10px] text-slate-400">
                  {terrain ? `${terrain.elevation_m}m ASL | ${terrain.slope_degrees}° Slope` : 'Global Elevation'}
                </span>
              </div>
            </div>
            <div className="flex items-center gap-1 text-[11px]">
              {terrain ? (
                <span className="text-emerald-400 flex items-center gap-1 font-mono">
                  <CheckCircle2 className="w-3.5 h-3.5" /> Ready
                </span>
              ) : (
                <span className="text-slate-500 flex items-center gap-1 font-mono">
                  <AlertCircle className="w-3.5 h-3.5" /> Pending
                </span>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* 4. Rainfall Summary Cards */}
      {weather && (
        <div className="grid grid-cols-2 gap-2 text-xs">
          <div className="p-3 bg-slate-950 rounded-xl border border-slate-800">
            <span className="text-[10px] text-slate-400 uppercase block mb-1">Past 7-Day Rainfall</span>
            <span className="text-base font-bold text-sky-400 font-mono">
              {weather.summary.past_7d_total_rain_mm} mm
            </span>
          </div>
          <div className="p-3 bg-slate-950 rounded-xl border border-slate-800">
            <span className="text-[10px] text-slate-400 uppercase block mb-1">Next 48h Forecast</span>
            <span className="text-base font-bold text-indigo-400 font-mono">
              {weather.summary.forecast_48h_total_rain_mm} mm
            </span>
          </div>
        </div>
      )}

      {/* 5. Key Prediction Factor Explanations */}
      {prediction?.factor_importances && prediction.factor_importances.length > 0 && (
        <div>
          <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">
            Key Predictive Drivers
          </h4>
          <div className="space-y-1.5">
            {prediction.factor_importances.map((fact, idx) => {
              const ImpactIcon =
                fact.impact === 'increasing_risk'
                  ? TrendingUp
                  : fact.impact === 'decreasing_risk'
                  ? TrendingDown
                  : Minus;
              const impactColor =
                fact.impact === 'increasing_risk'
                  ? 'text-rose-400'
                  : fact.impact === 'decreasing_risk'
                  ? 'text-emerald-400'
                  : 'text-slate-400';

              return (
                <div key={idx} className="p-2 bg-slate-950 rounded-lg border border-slate-800/80 text-xs flex items-center justify-between">
                  <div className="flex items-center gap-1.5">
                    <ImpactIcon className={`w-3.5 h-3.5 ${impactColor}`} />
                    <span className="text-slate-300 font-medium">{fact.factor_name}</span>
                  </div>
                  <div className="text-right">
                    <span className="font-mono text-slate-200">
                      {fact.value} {fact.unit}
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* 6. Export Results */}
      <div className="pt-2 border-t border-slate-800 flex gap-2">
        <button
          onClick={downloadCSV}
          disabled={!prediction}
          className="flex-1 py-2 px-3 rounded-lg bg-slate-800 hover:bg-slate-700 disabled:opacity-40 text-xs font-medium text-slate-300 flex items-center justify-center gap-1.5 transition-colors"
        >
          <Download className="w-3.5 h-3.5" /> CSV
        </button>
        <button
          onClick={downloadJSON}
          disabled={!prediction}
          className="flex-1 py-2 px-3 rounded-lg bg-slate-800 hover:bg-slate-700 disabled:opacity-40 text-xs font-medium text-slate-300 flex items-center justify-center gap-1.5 transition-colors"
        >
          <Download className="w-3.5 h-3.5" /> JSON
        </button>
      </div>

      {/* 7. Research Disclaimer */}
      <div className="text-[10px] text-slate-400 leading-relaxed p-2.5 rounded-lg bg-slate-950/60 border border-slate-800/60 flex items-start gap-1.5">
        <Info className="w-3.5 h-3.5 text-cyan-500 shrink-0 mt-0.5" />
        <span>
          Academic research model. Predictions are experimental probabilities and DO NOT substitute for official civil protection or meteorological warnings.
        </span>
      </div>
    </div>
  );
};
