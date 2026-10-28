from typing import Optional
from pydantic import BaseModel, Field

class HourlyDataPoint(BaseModel):
    timestamp: str
    precipitation_mm: Optional[float] = None
    temperature_2m_c: Optional[float] = None
    relative_humidity_pct: Optional[float] = None
    surface_pressure_hpa: Optional[float] = None
    wind_speed_10m_kmh: Optional[float] = None
    soil_moisture_m3m3: Optional[float] = None

class WeatherSequence(BaseModel):
    data_type: str = Field(..., description="'historical_observation' or 'future_forecast'")
    start_time: str
    end_time: str
    hours_count: int
    hourly: list[HourlyDataPoint]
    total_precipitation_mm: float
    max_precipitation_mm: float
    avg_soil_moisture: Optional[float] = None

class WeatherResponse(BaseModel):
    status: str = "success"
    latitude: float
    longitude: float
    timezone: str
    elevation_m: Optional[float] = None
    retrieval_timestamp: str
    historical_observations_168h: WeatherSequence
    forecast_48h: WeatherSequence
    summary: dict
    data_source: str = "Open-Meteo Global Forecasting & Archive System"
    notes: list[str] = Field(default_factory=list)
