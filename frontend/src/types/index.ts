export type RiskCategory = 'low' | 'moderate' | 'high' | 'very_high' | 'unavailable';

export interface LocationSearchResult {
  id?: number;
  name: string;
  latitude: number;
  longitude: number;
  country?: string;
  country_code?: string;
  admin1?: string;
  elevation?: number;
  timezone?: string;
}

export interface HourlyDataPoint {
  timestamp: string;
  precipitation_mm?: number | null;
  temperature_2m_c?: number | null;
  relative_humidity_pct?: number | null;
  surface_pressure_hpa?: number | null;
  wind_speed_10m_kmh?: number | null;
  soil_moisture_m3m3?: number | null;
}

export interface WeatherSequence {
  data_type: 'historical_observation' | 'future_forecast';
  start_time: string;
  end_time: string;
  hours_count: number;
  hourly: HourlyDataPoint[];
  total_precipitation_mm: number;
  max_precipitation_mm: number;
  avg_soil_moisture?: number | null;
}

export interface WeatherResponse {
  status: string;
  latitude: number;
  longitude: number;
  timezone: string;
  elevation_m?: number | null;
  retrieval_timestamp: string;
  historical_observations_168h: WeatherSequence;
  forecast_48h: WeatherSequence;
  summary: {
    past_7d_total_rain_mm: number;
    forecast_48h_total_rain_mm: number;
    max_hourly_rain_mm: number;
    soil_saturation_proxy?: number | null;
    heavy_rain_warning: boolean;
  };
  data_source: string;
  notes: string[];
}

export interface SatelliteMetadata {
  satellite: string;
  platform: string;
  instrument: string;
  mode: string;
  polarizations: string[];
  product_type: string;
  resolution_m: number;
  patch_size_pixels: number;
  orbit_direction?: string | null;
  relative_orbit?: number | null;
  acquisition_time?: string | null;
  scene_id?: string | null;
}

export interface SatelliteResponse {
  status: 'available' | 'coverage_confirmed' | 'unavailable' | 'authentication_required';
  location: { latitude: number; longitude: number };
  data_source: string;
  metadata?: SatelliteMetadata | null;
  mean_backscatter_vv_db?: number | null;
  mean_backscatter_vh_db?: number | null;
  water_index_proxy?: number | null;
  access_requirements: string;
  disclaimer: string;
}

export interface TerrainResponse {
  elevation_m: number;
  slope_degrees: number;
  terrain_roughness_index: number;
  topographic_wetness_index: number;
  estimated_dist_to_drainage_km: number;
  is_lowland: boolean;
  elevation_source: string;
  data_freshness: string;
}

export interface HistoricalFloodEvent {
  event_id: string;
  country: string;
  region_name: string;
  latitude: number;
  longitude: number;
  start_date: string;
  end_date: string;
  duration_days: number;
  flooded_area_sq_km?: number | null;
  severity_class: number;
  main_cause: string;
  dead?: number | null;
  displaced?: number | null;
  source_database: string;
  distance_km?: number | null;
  historical_flood_within_48h_label: number;
}

export interface FloodHistoryResponse {
  status: string;
  query_location: { latitude: number; longitude: number };
  radius_km: number;
  total_events_found: number;
  events: HistoricalFloodEvent[];
  database_version: string;
  disclaimer: string;
}

export interface DataSourceStatus {
  name: string;
  modality: 'weather_forecast' | 'weather_historical' | 'satellite_sar' | 'geographic_terrain';
  status: 'available' | 'partial' | 'unavailable' | 'mock_demo';
  retrieval_timestamp?: string | null;
  latency_ms?: number | null;
  source_attribution: string;
}

export interface FactorImportance {
  factor_name: string;
  feature_group: 'weather' | 'terrain' | 'satellite';
  value: number;
  unit: string;
  impact: 'increasing_risk' | 'neutral' | 'decreasing_risk';
  relative_weight: number;
}

export interface UncertaintyInfo {
  confidence_score: number;
  epistemic_uncertainty: number;
  aleatoric_uncertainty: number;
  lower_bound_90ci: number;
  upper_bound_90ci: number;
}

export interface PredictResponse {
  status: 'success' | 'partial_modalities' | 'unavailable' | 'model_not_trained' | 'demo';
  location: { latitude: number; longitude: number };
  forecast_horizon_hours: number;
  flood_probability?: number | null;
  risk_category: RiskCategory;
  model_version: string;
  model_architecture: string;
  data_sources: DataSourceStatus[];
  missing_modalities: string[];
  uncertainty?: UncertaintyInfo | null;
  factor_importances: FactorImportance[];
  calibrated_thresholds: Record<string, [number, number]>;
  prediction_timestamp: string;
  disclaimer: string;
  limitations: string[];
}

export interface ModelStatus {
  is_model_trained: boolean;
  is_smoke_test: boolean | null;
  checkpoint_file: string | null;
  checkpoint_metadata: {
    epoch?: number;
    val_loss?: number;
    val_auc?: number;
    is_smoke_test?: boolean;
    architecture?: string;
  };
  training_history: {
    train_loss: number[];
    val_loss: number[];
    val_auc: number[];
    val_f1: number[];
  } | null;
  calibrator_fitted: boolean;
}

export interface ModelEvaluation {
  status: string;
  is_smoke_test?: boolean;
  sample_count: number;
  metrics: {
    roc_auc: number;
    pr_auc: number;
    f1_score: number;
    precision: number;
    recall: number;
    brier_score: number;
  } | null;
  confusion_matrix: number[][] | null;
  evaluation_notes: string[];
}

