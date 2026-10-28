import React, { useState, useEffect } from 'react';
import { SidebarControls } from '../components/SidebarControls';
import { MapComponent } from '../components/MapComponent';
import { PredictionResultsPanel } from '../components/PredictionResultsPanel';
import { WeatherChart } from '../components/WeatherChart';
import { ErrorBoundary } from '../components/ErrorBoundary';
import {
  fetchWeather,
  fetchSatellite,
  fetchTerrain,
  fetchFloodHistory,
  runPrediction
} from '../services/api';
import {
  WeatherResponse,
  SatelliteResponse,
  TerrainResponse,
  PredictResponse,
  HistoricalFloodEvent
} from '../types';
import { AlertCircle, CheckCircle } from 'lucide-react';

export const DashboardPage: React.FC = () => {
  // Default to Chennai, India (13.0827, 80.2707) as a rich historical benchmark location
  const [lat, setLat] = useState(13.0827);
  const [lon, setLon] = useState(80.2707);
  const [forecastHorizon, setForecastHorizon] = useState(48);

  const [weather, setWeather] = useState<WeatherResponse | null>(null);
  const [satellite, setSatellite] = useState<SatelliteResponse | null>(null);
  const [terrain, setTerrain] = useState<TerrainResponse | null>(null);
  const [historicalEvents, setHistoricalEvents] = useState<HistoricalFloodEvent[]>([]);
  const [prediction, setPrediction] = useState<PredictResponse | null>(null);

  const [isLoadingInputs, setIsLoadingInputs] = useState(false);
  const [isPredicting, setIsPredicting] = useState(false);
  const [demoMode, setDemoMode] = useState(false);

  // Layer switches
  const [showRiskOverlay, setShowRiskOverlay] = useState(true);
  const [showHistoricalOverlay, setShowHistoricalOverlay] = useState(true);
  const [showSatelliteCoverage, setShowSatelliteCoverage] = useState(true);

  const [notification, setNotification] = useState<{ type: 'info' | 'error' | 'success'; text: string } | null>(null);

  // Fetch contextual inputs when location changes
  useEffect(() => {
    let isCancelled = false;

    async function loadLocationData() {
      setIsLoadingInputs(true);
      try {
        const [w, s, t, h] = await Promise.allSettled([
          fetchWeather(lat, lon),
          fetchSatellite(lat, lon),
          fetchTerrain(lat, lon),
          fetchFloodHistory(lat, lon, 2000)
        ]);

        if (isCancelled) return;

        if (w.status === 'fulfilled') setWeather(w.value);
        if (s.status === 'fulfilled') setSatellite(s.value);
        if (t.status === 'fulfilled') setTerrain(t.value);
        if (h.status === 'fulfilled') setHistoricalEvents(h.value.events || []);

        setNotification({
          type: 'info',
          text: `Retrieved data for Lat: ${lat.toFixed(4)}, Lon: ${lon.toFixed(4)}`
        });
      } catch (err: any) {
        if (!isCancelled) {
          setNotification({ type: 'error', text: `Failed to load location inputs: ${err.message}` });
        }
      } finally {
        if (!isCancelled) setIsLoadingInputs(false);
      }
    }

    loadLocationData();
    return () => {
      isCancelled = true;
    };
  }, [lat, lon]);

  // Execute prediction
  const handleRunPrediction = async () => {
    setIsPredicting(true);
    try {
      const result = await runPrediction(lat, lon, forecastHorizon, demoMode);
      setPrediction(result);
      setNotification({
        type: 'success',
        text: `48-Hour Prediction generated (${result.risk_category.toUpperCase()} Risk)`
      });
    } catch (err: any) {
      setNotification({ type: 'error', text: `Prediction failed: ${err.message}` });
    } finally {
      setIsPredicting(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Dynamic Status / Notification Banner */}
      {notification && (
        <div
          className={`flex items-center justify-between px-4 py-2.5 rounded-xl border text-xs transition-all ${
            notification.type === 'success'
              ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-300'
              : notification.type === 'error'
              ? 'bg-rose-500/10 border-rose-500/30 text-rose-300'
              : 'bg-cyan-500/10 border-cyan-500/30 text-cyan-300'
          }`}
        >
          <div className="flex items-center gap-2">
            {notification.type === 'success' ? (
              <CheckCircle className="w-4 h-4 text-emerald-400" />
            ) : (
              <AlertCircle className="w-4 h-4 text-cyan-400" />
            )}
            <span>{notification.text}</span>
          </div>
          <button
            onClick={() => setNotification(null)}
            className="text-slate-400 hover:text-slate-200 ml-4 font-bold"
          >
            ✕
          </button>
        </div>
      )}

      {/* Main Grid: Left Controls (3 cols), Center Map (5 cols), Right Results (4 cols) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Left Sidebar Controls */}
        <div className="lg:col-span-3">
          <SidebarControls
            lat={lat}
            lon={lon}
            onCoordinatesChange={(newLat, newLon) => {
              setLat(newLat);
              setLon(newLon);
            }}
            forecastHorizon={forecastHorizon}
            setForecastHorizon={setForecastHorizon}
            onRunPrediction={handleRunPrediction}
            isPredicting={isPredicting}
            demoMode={demoMode}
            setDemoMode={setDemoMode}
            showRiskOverlay={showRiskOverlay}
            setShowRiskOverlay={setShowRiskOverlay}
            showHistoricalOverlay={showHistoricalOverlay}
            setShowHistoricalOverlay={setShowHistoricalOverlay}
            showSatelliteCoverage={showSatelliteCoverage}
            setShowSatelliteCoverage={setShowSatelliteCoverage}
          />
        </div>

        {/* Center Main Map */}
        <div className="lg:col-span-5 h-[580px]">
          <MapComponent
            selectedLat={lat}
            selectedLon={lon}
            onLocationSelect={(newLat, newLon) => {
              setLat(newLat);
              setLon(newLon);
            }}
            prediction={prediction}
            historicalEvents={historicalEvents}
            showRiskOverlay={showRiskOverlay}
            showHistoricalOverlay={showHistoricalOverlay}
            showSatelliteCoverage={showSatelliteCoverage}
          />
        </div>

        {/* Right Prediction Results Panel */}
        <div className="lg:col-span-4">
          <PredictionResultsPanel
            prediction={prediction}
            weather={weather}
            satellite={satellite}
            terrain={terrain}
            isLoading={isPredicting || isLoadingInputs}
          />
        </div>
      </div>

      {/* Weather Time-Series Chart */}
      <div className="w-full">
        <ErrorBoundary fallbackTitle="Weather Time-Series Chart Error">
          <WeatherChart weather={weather} />
        </ErrorBoundary>
      </div>
    </div>
  );
};
