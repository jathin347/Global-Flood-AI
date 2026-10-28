import {
  LocationSearchResult,
  WeatherResponse,
  SatelliteResponse,
  TerrainResponse,
  FloodHistoryResponse,
  PredictResponse,
  ModelStatus
} from '../types';

const API_BASE = '';

export async function searchLocations(query: string): Promise<LocationSearchResult[]> {
  if (!query.trim()) return [];
  const res = await fetch(`${API_BASE}/api/location/search?q=${encodeURIComponent(query)}`);
  if (!res.ok) throw new Error(`Location search failed: ${res.statusText}`);
  const data = await res.json();
  return data.results || [];
}

export async function fetchWeather(lat: number, lon: number): Promise<WeatherResponse> {
  const res = await fetch(`${API_BASE}/api/weather?lat=${lat}&lon=${lon}`);
  if (!res.ok) throw new Error(`Weather fetch failed: ${res.statusText}`);
  return res.json();
}

export async function fetchSatellite(lat: number, lon: number): Promise<SatelliteResponse> {
  const res = await fetch(`${API_BASE}/api/satellite?lat=${lat}&lon=${lon}`);
  if (!res.ok) throw new Error(`Satellite query failed: ${res.statusText}`);
  return res.json();
}

export async function fetchTerrain(lat: number, lon: number): Promise<TerrainResponse> {
  const res = await fetch(`${API_BASE}/api/terrain?lat=${lat}&lon=${lon}`);
  if (!res.ok) throw new Error(`Terrain query failed: ${res.statusText}`);
  return res.json();
}

export async function fetchFloodHistory(lat: number, lon: number, radiusKm: number = 2000): Promise<FloodHistoryResponse> {
  const res = await fetch(`${API_BASE}/api/flood/history?lat=${lat}&lon=${lon}&radius_km=${radiusKm}`);
  if (!res.ok) throw new Error(`Flood history failed: ${res.statusText}`);
  return res.json();
}

export async function runPrediction(
  lat: number,
  lon: number,
  horizon: number = 48,
  demoMode: boolean = false
): Promise<PredictResponse> {
  const res = await fetch(`${API_BASE}/api/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      latitude: lat,
      longitude: lon,
      forecast_horizon_hours: horizon,
      simulation_demo_mode: demoMode
    })
  });
  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || `Prediction failed: ${res.statusText}`);
  }
  return res.json();
}

export async function fetchModelStatus(): Promise<ModelStatus> {
  const res = await fetch(`${API_BASE}/api/model/status`);
  if (!res.ok) throw new Error(`Model status failed: ${res.statusText}`);
  return res.json();
}

export async function fetchModelEvaluation(): Promise<import('../types').ModelEvaluation> {
  const res = await fetch(`${API_BASE}/api/model/evaluation`);
  if (!res.ok) throw new Error(`Model evaluation failed: ${res.statusText}`);
  return res.json();
}

export async function checkBackendHealth(): Promise<any> {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error('Backend unhealthy');
  return res.json();
}

