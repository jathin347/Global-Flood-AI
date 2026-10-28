import httpx
import numpy as np
from app.core.config import settings

def fetch_historical_weather_archive(
    lat: float,
    lon: float,
    start_date: str,  # YYYY-MM-DD
    end_date: str
) -> dict:
    """
    Retrieves reanalysis weather data from Open-Meteo Historical Archive API.
    Preserves what information would have been available at the prediction time.
    """
    url = settings.OPEN_METEO_HISTORICAL_URL
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": [
            "precipitation",
            "temperature_2m",
            "relative_humidity_2m",
            "surface_pressure",
            "wind_speed_10m",
            "soil_moisture_0_to_1cm"
        ],
        "timezone": "UTC"
    }
    with httpx.Client(timeout=30.0) as client:
        resp = client.get(url, params=params)
        resp.raise_for_status()
        return resp.json()


def create_smoke_test_dataset(num_samples: int = 40) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Generates a reproducible synthetic dataset strictly for SMOKE-TESTING
    tensor shapes, gradients, fusion layers, and pipeline mechanics.
    WARNING: Synthetic samples MUST NEVER be used to evaluate real model accuracy.
    """
    np.random.seed(42)
    samples = []
    
    # 4 distinct synthetic flood events to simulate spatial/event grouping
    event_ids = ["SMOKE_EVENT_A", "SMOKE_EVENT_B", "SMOKE_EVENT_C", "SMOKE_EVENT_D"]
    
    samples_per_event = max(1, num_samples // len(event_ids))
    for i in range(num_samples):
        event_idx = min(len(event_ids) - 1, i // samples_per_event)
        event_id = event_ids[event_idx]
        is_flood = 1.0 if i % 2 == 0 else 0.0

        # Synthetic SAR backscatter patch (2, 128, 128)
        # Flooded water surfaces have very low radar backscatter (-25 dB to -18 dB)
        base_db = -22.0 if is_flood == 1.0 else -12.0
        sar_patch = np.random.normal(loc=base_db, scale=2.5, size=(2, 128, 128)).astype(np.float32)

        # Weather sequences: 168h historical + 48h forecast
        hist_weather = []
        for h in range(168):
            rain = max(0.0, np.random.exponential(scale=3.5 if is_flood else 0.4))
            hist_weather.append({
                "precipitation_mm": round(rain, 2),
                "temperature_2m_c": round(22.0 + np.sin(h / 24.0) * 5.0, 1),
                "relative_humidity_pct": round(min(100.0, 75.0 + (15.0 if is_flood else 0.0)), 1),
                "surface_pressure_hpa": round(1008.0 - (12.0 if is_flood else 0.0) + np.random.normal(0, 1), 1),
                "wind_speed_10m_kmh": round(max(0.0, 15.0 + (15.0 if is_flood else 0.0)), 1),
                "soil_moisture_m3m3": round(min(0.5, 0.35 + (0.1 if is_flood else 0.0)), 3)
            })

        fc_weather = []
        for h in range(48):
            fc_rain = max(0.0, np.random.exponential(scale=5.0 if is_flood else 0.2))
            fc_weather.append({
                "precipitation_mm": round(fc_rain, 2),
                "temperature_2m_c": round(21.0 + np.sin(h / 24.0) * 5.0, 1),
                "relative_humidity_pct": round(min(100.0, 80.0 + (15.0 if is_flood else 0.0)), 1),
                "surface_pressure_hpa": round(1005.0 - (15.0 if is_flood else 0.0) + np.random.normal(0, 1), 1),
                "wind_speed_10m_kmh": round(max(0.0, 18.0 + (10.0 if is_flood else 0.0)), 1),
                "soil_moisture_m3m3": None
            })

        terrain = {
            "elevation_m": round(max(1.0, 12.0 if is_flood else 85.0 + np.random.normal(0, 15)), 1),
            "slope_degrees": round(max(0.2, 1.2 if is_flood else 6.5 + np.random.normal(0, 2)), 1),
            "terrain_roughness_index": round(max(0.5, 3.0 if is_flood else 9.0), 1),
            "topographic_wetness_index": round(10.5 if is_flood else 5.5, 1),
            "estimated_dist_to_drainage_km": round(max(0.2, 1.5 if is_flood else 15.0), 1)
        }

        samples.append({
            "sample_id": f"smoke_sample_{i:04d}",
            "event_id": event_id,
            "latitude": round(13.08 + (i * 0.05), 4),
            "longitude": round(80.27 + (i * 0.05), 4),
            "prediction_timestamp": "2024-10-01T00:00:00Z",
            "sar_patch": sar_patch.tolist(),
            "historical_weather_168h": hist_weather,
            "forecast_weather_48h": fc_weather,
            "terrain_features": terrain,
            "target_flood_within_48h": is_flood,
            "provenance": "smoke_test_synthetic",
            "is_smoke_test": True
        })

    # Event-grouped split to prevent leakage
    train_samples = [s for s in samples if s["event_id"] in ["SMOKE_EVENT_A", "SMOKE_EVENT_B"]]
    val_samples = [s for s in samples if s["event_id"] == "SMOKE_EVENT_C"]
    test_samples = [s for s in samples if s["event_id"] == "SMOKE_EVENT_D"]

    return train_samples, val_samples, test_samples
