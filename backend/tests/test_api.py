import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "device" in data
    assert "model_loaded" in data

def test_location_search():
    response = client.get("/api/location/search?q=Munich")
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "Munich"
    assert len(data["results"]) > 0
    assert "latitude" in data["results"][0]
    assert "longitude" in data["results"][0]

def test_weather_endpoint_separation():
    # Test coordinates (Chennai, India)
    response = client.get("/api/weather?lat=13.0827&lon=80.2707")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    # Verify strict separation of historical observations (168h) vs forecast (48h)
    assert data["historical_observations_168h"]["data_type"] == "historical_observation"
    assert data["forecast_48h"]["data_type"] == "future_forecast"
    assert len(data["historical_observations_168h"]["hourly"]) > 0
    assert len(data["forecast_48h"]["hourly"]) > 0

def test_satellite_endpoint():
    response = client.get("/api/satellite?lat=13.0827&lon=80.2707")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["available", "coverage_confirmed", "unavailable"]
    assert "Copernicus" in data["data_source"]

def test_terrain_endpoint():
    response = client.get("/api/terrain?lat=13.0827&lon=80.2707")
    assert response.status_code == 200
    data = response.json()
    assert "elevation_m" in data
    assert "slope_degrees" in data

def test_flood_history_endpoint():
    response = client.get("/api/flood/history?lat=13.0827&lon=80.2707&radius_km=1000")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["total_events_found"] > 0
    # Chennai 2015 event should be found
    event_ids = [e["event_id"] for e in data["events"]]
    assert "GFD_MODIS_3914" in event_ids

def test_predict_endpoint():
    payload = {
        "latitude": 13.0827,
        "longitude": 80.2707,
        "forecast_horizon_hours": 48,
        "simulation_demo_mode": False
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["success", "partial_modalities", "model_not_trained"]
    assert data["forecast_horizon_hours"] == 48
    assert "risk_category" in data
    assert len(data["data_sources"]) > 0
    assert "disclaimer" in data
    assert "RESEARCH DISCLAIMER" in data["disclaimer"]

def test_invalid_coordinates_weather():
    response = client.get("/api/weather?lat=999.0&lon=80.0")
    assert response.status_code == 422

def test_invalid_coordinates_predict():
    payload = {
        "latitude": 105.0,
        "longitude": 80.2707,
        "forecast_horizon_hours": 48
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 422

def test_model_status_endpoint():
    response = client.get("/api/model/status")
    assert response.status_code == 200
    data = response.json()
    assert "is_model_trained" in data
    assert "checkpoint_file" in data

def test_predict_demo_mode():
    payload = {
        "latitude": 13.0827,
        "longitude": 80.2707,
        "forecast_horizon_hours": 48,
        "simulation_demo_mode": True
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["demo", "success", "partial_modalities"]
    assert data["flood_probability"] is not None
    assert data["risk_category"] in ["low", "moderate", "high", "very_high"]

