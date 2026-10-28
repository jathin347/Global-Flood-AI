import httpx
import time
from datetime import datetime, timezone
from app.core.config import settings
from app.core.logging import logger
from app.schemas.weather import WeatherResponse, WeatherSequence, HourlyDataPoint

class WeatherService:
    def __init__(self):
        self.timeout = settings.HTTP_TIMEOUT_SECONDS
        self._cache: dict[str, tuple[float, WeatherResponse]] = {}

    def _get_cache_key(self, lat: float, lon: float) -> str:
        # Cache key rounded to ~1km resolution (0.01 degree)
        return f"{round(lat, 2)}_{round(lon, 2)}"

    async def get_weather(self, lat: float, lon: float) -> WeatherResponse:
        cache_key = self._get_cache_key(lat, lon)
        now = time.time()
        if cache_key in self._cache:
            cache_time, cached_data = self._cache[cache_key]
            if now - cache_time < settings.CACHE_EXPIRY_SECONDS:
                logger.info(f"Returning cached weather for {lat}, {lon}")
                return cached_data

        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": [
                "precipitation",
                "temperature_2m",
                "relative_humidity_2m",
                "surface_pressure",
                "wind_speed_10m",
                "soil_moisture_0_to_1cm"
            ],
            "past_days": 7,
            "forecast_days": 3,
            "timezone": "auto"
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                resp = await client.get(settings.OPEN_METEO_FORECAST_URL, params=params)
                resp.raise_for_status()
                data = resp.json()
            except httpx.HTTPStatusError as e:
                logger.error(f"Open-Meteo API HTTP error {e.response.status_code}: {e}")
                raise ValueError(f"Weather API returned status {e.response.status_code}: {e.response.text}")
            except Exception as e:
                logger.error(f"Open-Meteo API connection error: {e}")
                raise RuntimeError(f"Unable to reach Open-Meteo weather service: {str(e)}")

        hourly = data.get("hourly", {})
        times = hourly.get("time", [])
        precip = hourly.get("precipitation", [])
        temp = hourly.get("temperature_2m", [])
        humidity = hourly.get("relative_humidity_2m", [])
        pressure = hourly.get("surface_pressure", [])
        wind = hourly.get("wind_speed_10m", [])
        soil = hourly.get("soil_moisture_0_to_1cm", [])

        # Partition into historical observations (past 168 hours) vs future forecast (next 48 hours)
        # Open-Meteo returns past 7 days (7 * 24 = 168 hours) starting at index 0.
        # Find the cutoff: current hour in the location's local time or index 168
        current_time_str = datetime.now(timezone.utc).isoformat()
        
        # Typically past_days=7 means first 168 elements are past observations
        hist_cutoff = min(168, len(times))
        forecast_end = min(hist_cutoff + 48, len(times))

        historical_points: list[HourlyDataPoint] = []
        for i in range(hist_cutoff):
            historical_points.append(
                HourlyDataPoint(
                    timestamp=times[i],
                    precipitation_mm=precip[i] if i < len(precip) else None,
                    temperature_2m_c=temp[i] if i < len(temp) else None,
                    relative_humidity_pct=humidity[i] if i < len(humidity) else None,
                    surface_pressure_hpa=pressure[i] if i < len(pressure) else None,
                    wind_speed_10m_kmh=wind[i] if i < len(wind) else None,
                    soil_moisture_m3m3=soil[i] if (soil and i < len(soil)) else None
                )
            )

        forecast_points: list[HourlyDataPoint] = []
        for i in range(hist_cutoff, forecast_end):
            forecast_points.append(
                HourlyDataPoint(
                    timestamp=times[i],
                    precipitation_mm=precip[i] if i < len(precip) else None,
                    temperature_2m_c=temp[i] if i < len(temp) else None,
                    relative_humidity_pct=humidity[i] if i < len(humidity) else None,
                    surface_pressure_hpa=pressure[i] if i < len(pressure) else None,
                    wind_speed_10m_kmh=wind[i] if i < len(wind) else None,
                    soil_moisture_m3m3=soil[i] if (soil and i < len(soil)) else None
                )
            )

        # Compute summary metrics
        hist_valid_precip = [p.precipitation_mm for p in historical_points if p.precipitation_mm is not None]
        fc_valid_precip = [p.precipitation_mm for p in forecast_points if p.precipitation_mm is not None]
        hist_valid_soil = [p.soil_moisture_m3m3 for p in historical_points if p.soil_moisture_m3m3 is not None]

        hist_seq = WeatherSequence(
            data_type="historical_observation",
            start_time=times[0] if times else "",
            end_time=times[hist_cutoff - 1] if hist_cutoff > 0 else "",
            hours_count=len(historical_points),
            hourly=historical_points,
            total_precipitation_mm=round(sum(hist_valid_precip), 2) if hist_valid_precip else 0.0,
            max_precipitation_mm=round(max(hist_valid_precip), 2) if hist_valid_precip else 0.0,
            avg_soil_moisture=round(sum(hist_valid_soil) / len(hist_valid_soil), 3) if hist_valid_soil else None
        )

        fc_seq = WeatherSequence(
            data_type="future_forecast",
            start_time=times[hist_cutoff] if hist_cutoff < len(times) else "",
            end_time=times[forecast_end - 1] if forecast_end > hist_cutoff else "",
            hours_count=len(forecast_points),
            hourly=forecast_points,
            total_precipitation_mm=round(sum(fc_valid_precip), 2) if fc_valid_precip else 0.0,
            max_precipitation_mm=round(max(fc_valid_precip), 2) if fc_valid_precip else 0.0,
            avg_soil_moisture=None
        )

        response = WeatherResponse(
            status="success",
            latitude=data.get("latitude", lat),
            longitude=data.get("longitude", lon),
            timezone=data.get("timezone", "UTC"),
            elevation_m=data.get("elevation"),
            retrieval_timestamp=current_time_str,
            historical_observations_168h=hist_seq,
            forecast_48h=fc_seq,
            summary={
                "past_7d_total_rain_mm": hist_seq.total_precipitation_mm,
                "forecast_48h_total_rain_mm": fc_seq.total_precipitation_mm,
                "max_hourly_rain_mm": max(hist_seq.max_precipitation_mm, fc_seq.max_precipitation_mm),
                "soil_saturation_proxy": hist_seq.avg_soil_moisture,
                "heavy_rain_warning": fc_seq.total_precipitation_mm > 50.0
            },
            data_source="Open-Meteo Global Numerical Weather Prediction Model Ensemble",
            notes=[
                "Observations and forecasts are rigorously kept in separate sequences to avoid future-leakage.",
                "Soil moisture represents topsoil 0-1cm depth volumetric water content."
            ]
        )

        self._cache[cache_key] = (now, response)
        return response

weather_service = WeatherService()
