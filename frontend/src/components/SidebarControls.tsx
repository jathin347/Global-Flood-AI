import React, { useState, useEffect } from 'react';
import { Search, Play, Layers, Compass, Sparkles } from 'lucide-react';
import { searchLocations } from '../services/api';
import { LocationSearchResult } from '../types';

interface SidebarControlsProps {
  lat: number;
  lon: number;
  onCoordinatesChange: (lat: number, lon: number) => void;
  forecastHorizon: number;
  setForecastHorizon: (h: number) => void;
  onRunPrediction: () => void;
  isPredicting: boolean;
  demoMode: boolean;
  setDemoMode: (d: boolean) => void;
  showRiskOverlay: boolean;
  setShowRiskOverlay: (v: boolean) => void;
  showHistoricalOverlay: boolean;
  setShowHistoricalOverlay: (v: boolean) => void;
  showSatelliteCoverage: boolean;
  setShowSatelliteCoverage: (v: boolean) => void;
}

const PRESET_LOCATIONS = [
  { name: 'Chennai, India', lat: 13.0827, lon: 80.2707, tag: 'Coastal Monsoon' },
  { name: 'Ahr Valley, Germany', lat: 50.5438, lon: 7.1136, tag: 'Flash Flood' },
  { name: 'Sindh, Pakistan', lat: 27.5589, lon: 68.2020, tag: 'Riverine Basin' },
  { name: 'Sylhet, Bangladesh', lat: 24.8949, lon: 91.8687, tag: 'Catchment Runoff' },
  { name: 'New Orleans, USA', lat: 29.9511, lon: -90.0715, tag: 'Lowland Delta' }
];

export const SidebarControls: React.FC<SidebarControlsProps> = ({
  lat,
  lon,
  onCoordinatesChange,
  forecastHorizon,
  setForecastHorizon,
  onRunPrediction,
  isPredicting,
  demoMode,
  setDemoMode,
  showRiskOverlay,
  setShowRiskOverlay,
  showHistoricalOverlay,
  setShowHistoricalOverlay,
  showSatelliteCoverage,
  setShowSatelliteCoverage
}) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<LocationSearchResult[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [inputLat, setInputLat] = useState(lat.toString());
  const [inputLon, setInputLon] = useState(lon.toString());

  // Update input text when external coordinates change
  useEffect(() => {
    setInputLat(lat.toFixed(4));
    setInputLon(lon.toFixed(4));
  }, [lat, lon]);

  // Debounced location search
  useEffect(() => {
    if (!searchQuery.trim() || searchQuery.length < 2) {
      setSearchResults([]);
      return;
    }

    const timer = setTimeout(async () => {
      setIsSearching(true);
      try {
        const results = await searchLocations(searchQuery);
        setSearchResults(results);
      } catch (err) {
        console.error(err);
      } finally {
        setIsSearching(false);
      }
    }, 350);

    return () => clearTimeout(timer);
  }, [searchQuery]);

  const handleManualCoordSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const parsedLat = parseFloat(inputLat);
    const parsedLon = parseFloat(inputLon);
    if (!isNaN(parsedLat) && !isNaN(parsedLon) && parsedLat >= -90 && parsedLat <= 90 && parsedLon >= -180 && parsedLon <= 180) {
      onCoordinatesChange(parsedLat, parsedLon);
    }
  };

  const handleSelectPreset = (pLat: number, pLon: number) => {
    onCoordinatesChange(pLat, pLon);
  };

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl flex flex-col gap-6 text-slate-200">
      {/* 1. Global Location Search */}
      <div>
        <label className="text-xs font-semibold uppercase tracking-wider text-slate-400 block mb-2 flex items-center gap-1.5">
          <Search className="w-3.5 h-3.5 text-cyan-400" />
          Global Location Search
        </label>
        <div className="relative">
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search city, basin, or country..."
            className="w-full bg-slate-950 border border-slate-700 rounded-xl px-4 py-2.5 text-sm text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500 transition-colors"
          />
          {isSearching && (
            <div className="absolute right-3 top-3 w-4 h-4 border-2 border-cyan-500 border-t-transparent rounded-full animate-spin"></div>
          )}
          {searchResults.length > 0 && (
            <div className="absolute z-30 mt-1 w-full bg-slate-950 border border-slate-700 rounded-xl shadow-2xl overflow-hidden max-h-56 overflow-y-auto">
              {searchResults.map((item, idx) => (
                <button
                  key={idx}
                  type="button"
                  onClick={() => {
                    onCoordinatesChange(item.latitude, item.longitude);
                    setSearchQuery(`${item.name}${item.country ? ', ' + item.country : ''}`);
                    setSearchResults([]);
                  }}
                  className="w-full text-left px-3.5 py-2.5 text-xs hover:bg-slate-800 border-b border-slate-800/60 transition-colors flex items-center justify-between"
                >
                  <span className="font-medium text-slate-100 truncate">
                    {item.name} {item.admin1 ? `(${item.admin1})` : ''}
                  </span>
                  <span className="text-slate-400 font-mono text-[11px] ml-2 shrink-0">
                    {item.country || 'Global'}
                  </span>
                </button>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* 2. Manual Latitude & Longitude */}
      <form onSubmit={handleManualCoordSubmit} className="space-y-2">
        <label className="text-xs font-semibold uppercase tracking-wider text-slate-400 block flex items-center gap-1.5">
          <Compass className="w-3.5 h-3.5 text-cyan-400" />
          Geographic Coordinates
        </label>
        <div className="grid grid-cols-2 gap-2">
          <div>
            <span className="text-[10px] text-slate-400 uppercase font-mono">Latitude</span>
            <input
              type="number"
              step="any"
              value={inputLat}
              onChange={(e) => setInputLat(e.target.value)}
              className="w-full bg-slate-950 border border-slate-700 rounded-lg px-2.5 py-1.5 text-xs text-white font-mono focus:border-cyan-500 focus:outline-none"
            />
          </div>
          <div>
            <span className="text-[10px] text-slate-400 uppercase font-mono">Longitude</span>
            <input
              type="number"
              step="any"
              value={inputLon}
              onChange={(e) => setInputLon(e.target.value)}
              className="w-full bg-slate-950 border border-slate-700 rounded-lg px-2.5 py-1.5 text-xs text-white font-mono focus:border-cyan-500 focus:outline-none"
            />
          </div>
        </div>
        <button
          type="submit"
          className="w-full py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs font-medium text-slate-300 border border-slate-700 transition-colors"
        >
          Update Location Pin
        </button>
      </form>

      {/* 3. Quick Benchmark Presets */}
      <div>
        <label className="text-xs font-semibold uppercase tracking-wider text-slate-400 block mb-2">
          Historical Benchmark Locations
        </label>
        <div className="flex flex-wrap gap-1.5">
          {PRESET_LOCATIONS.map((preset) => (
            <button
              key={preset.name}
              type="button"
              onClick={() => handleSelectPreset(preset.lat, preset.lon)}
              className="px-2.5 py-1 rounded-md bg-slate-950 hover:bg-slate-800 border border-slate-800 text-[11px] text-slate-300 hover:text-cyan-400 transition-colors"
            >
              {preset.name}
            </button>
          ))}
        </div>
      </div>

      {/* 4. Forecast Horizon Selector */}
      <div>
        <label className="text-xs font-semibold uppercase tracking-wider text-slate-400 block mb-2">
          Prediction Horizon
        </label>
        <div className="grid grid-cols-3 gap-2">
          {[24, 48, 72].map((hours) => (
            <button
              key={hours}
              type="button"
              onClick={() => setForecastHorizon(hours)}
              className={`py-2 text-xs font-semibold rounded-xl border transition-all ${
                forecastHorizon === hours
                  ? 'bg-cyan-500/20 text-cyan-400 border-cyan-500 shadow-sm shadow-cyan-500/20'
                  : 'bg-slate-950 text-slate-400 border-slate-800 hover:border-slate-700'
              }`}
            >
              {hours} Hours
            </button>
          ))}
        </div>
      </div>

      {/* 5. Layer Toggles */}
      <div className="pt-2 border-t border-slate-800 space-y-2.5">
        <label className="text-xs font-semibold uppercase tracking-wider text-slate-400 block flex items-center gap-1.5">
          <Layers className="w-3.5 h-3.5 text-cyan-400" />
          Map Layer Controls
        </label>
        <label className="flex items-center justify-between text-xs cursor-pointer">
          <span className="text-slate-300">Model Risk Footprint (48h)</span>
          <input
            type="checkbox"
            checked={showRiskOverlay}
            onChange={(e) => setShowRiskOverlay(e.target.checked)}
            className="w-4 h-4 rounded bg-slate-900 border-slate-700 text-cyan-500 focus:ring-0"
          />
        </label>
        <label className="flex items-center justify-between text-xs cursor-pointer">
          <span className="text-slate-300">Historical Flood Events</span>
          <input
            type="checkbox"
            checked={showHistoricalOverlay}
            onChange={(e) => setShowHistoricalOverlay(e.target.checked)}
            className="w-4 h-4 rounded bg-slate-900 border-slate-700 text-rose-500 focus:ring-0"
          />
        </label>
        <label className="flex items-center justify-between text-xs cursor-pointer">
          <span className="text-slate-300">Sentinel-1 SAR Swath</span>
          <input
            type="checkbox"
            checked={showSatelliteCoverage}
            onChange={(e) => setShowSatelliteCoverage(e.target.checked)}
            className="w-4 h-4 rounded bg-slate-900 border-slate-700 text-cyan-500 focus:ring-0"
          />
        </label>
      </div>

      {/* 6. Simulation Demo Mode Switch */}
      <div className="pt-2 border-t border-slate-800">
        <div className="bg-amber-950/20 border border-amber-500/30 rounded-xl p-3">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-amber-300 flex items-center gap-1.5">
              <Sparkles className="w-3.5 h-3.5 text-amber-400" />
              Simulation Walkthrough Mode
            </span>
            <input
              type="checkbox"
              checked={demoMode}
              onChange={(e) => setDemoMode(e.target.checked)}
              className="w-4 h-4 rounded bg-slate-900 border-amber-600 text-amber-500 focus:ring-0"
            />
          </div>
          <p className="text-[11px] text-amber-400/80 mt-1">
            Enables full simulated prediction pipeline even if satellite raw swath requires CDSE token.
          </p>
        </div>
      </div>

      {/* 7. Action Button */}
      <button
        onClick={onRunPrediction}
        disabled={isPredicting}
        className="w-full mt-auto py-3.5 px-4 rounded-xl font-bold text-sm text-white bg-gradient-to-r from-cyan-600 via-blue-600 to-indigo-600 hover:from-cyan-500 hover:to-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed shadow-lg shadow-cyan-600/25 flex items-center justify-center gap-2 transition-all transform active:scale-[0.98]"
      >
        {isPredicting ? (
          <>
            <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
            <span>Evaluating Multimodal Features...</span>
          </>
        ) : (
          <>
            <Play className="w-4 h-4 fill-white" />
            <span>Generate 48-Hour Forecast</span>
          </>
        )}
      </button>
    </div>
  );
};
