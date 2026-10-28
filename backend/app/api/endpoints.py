import json
import torch
from fastapi import APIRouter, HTTPException, Query
from app.core.config import settings
from app.core.logging import logger
from app.schemas.location import LocationSearchResponse
from app.schemas.weather import WeatherResponse
from app.schemas.satellite import SatelliteResponse
from app.schemas.flood import FloodHistoryResponse
from app.schemas.prediction import PredictRequest, PredictResponse
from app.services.geocoding_service import geocoding_service
from app.services.weather_service import weather_service
from app.services.satellite_service import satellite_service
from app.services.terrain_service import terrain_service
from app.services.historical_flood_service import historical_flood_service
from ml.src.predict import predictor

router = APIRouter()

@router.get("/health")
async def health_check():
    """
    Health check endpoint returning service operational status,
    active ML device, and model checkpoint availability.
    """
    device_name = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    return {
        "status": "healthy",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "device": device_name,
        "model_loaded": predictor.is_loaded,
        "model_path": str(settings.MODEL_WEIGHTS_FILE),
        "is_smoke_test_weights": predictor.is_smoke_test if predictor.is_loaded else None
    }

@router.get("/api/location/search", response_model=LocationSearchResponse)
async def search_location(q: str = Query(..., min_length=1, description="Location search query")):
    """
    Geocodes a place name to global latitude and longitude coordinates.
    """
    try:
        return await geocoding_service.search(query=q)
    except Exception as e:
        logger.error(f"Error searching location for '{q}': {e}")
        raise HTTPException(status_code=500, detail=f"Location search failed: {str(e)}")

@router.get("/api/weather", response_model=WeatherResponse)
async def get_weather(
    lat: float = Query(..., ge=-90.0, le=90.0, description="Latitude"),
    lon: float = Query(..., ge=-180.0, le=180.0, description="Longitude")
):
    """
    Retrieves Open-Meteo 168h historical weather observations and 48h future numerical weather forecast.
    """
    try:
        return await weather_service.get_weather(lat=lat, lon=lon)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error fetching weather for ({lat}, {lon}): {e}")
        raise HTTPException(status_code=502, detail=f"Weather service error: {str(e)}")

@router.get("/api/satellite", response_model=SatelliteResponse)
async def get_satellite_info(
    lat: float = Query(..., ge=-90.0, le=90.0, description="Latitude"),
    lon: float = Query(..., ge=-180.0, le=180.0, description="Longitude")
):
    """
    Queries Copernicus Sentinel-1 Synthetic Aperture Radar (SAR) acquisition catalog and coverage.
    """
    try:
        return await satellite_service.get_sentinel1_metadata(lat=lat, lon=lon)
    except Exception as e:
        logger.error(f"Error querying satellite metadata for ({lat}, {lon}): {e}")
        raise HTTPException(status_code=500, detail=f"Satellite service error: {str(e)}")

@router.get("/api/terrain")
async def get_terrain(
    lat: float = Query(..., ge=-90.0, le=90.0, description="Latitude"),
    lon: float = Query(..., ge=-180.0, le=180.0, description="Longitude")
):
    """
    Extracts elevation, slope, terrain roughness index, and wetness index for coordinates.
    """
    try:
        return await terrain_service.get_terrain_features(lat=lat, lon=lon)
    except Exception as e:
        logger.error(f"Error extracting terrain for ({lat}, {lon}): {e}")
        raise HTTPException(status_code=500, detail=f"Terrain service error: {str(e)}")

@router.get("/api/flood/history", response_model=FloodHistoryResponse)
async def get_flood_history(
    lat: float = Query(..., ge=-90.0, le=90.0, description="Latitude"),
    lon: float = Query(..., ge=-180.0, le=180.0, description="Longitude"),
    radius_km: float = Query(default=1500.0, ge=50.0, le=10000.0, description="Proximity search radius in km")
):
    """
    Queries documented historical flood events from Global Flood DB (MODIS Events V1) and DFO.
    """
    try:
        return historical_flood_service.get_events_near(lat=lat, lon=lon, radius_km=radius_km)
    except Exception as e:
        logger.error(f"Error querying flood history for ({lat}, {lon}): {e}")
        raise HTTPException(status_code=500, detail=f"Historical flood service error: {str(e)}")

@router.get("/api/model/status")
async def get_model_status():
    """
    Returns active ML model status, checkpoint presence, and training metadata.
    """
    weights_path = settings.MODEL_WEIGHTS_FILE
    has_weights = weights_path.exists()
    ckpt_meta = {}
    if has_weights:
        try:
            device = torch.device("cpu")
            ckpt = torch.load(weights_path, map_location=device)
            ckpt_meta = {
                "epoch": ckpt.get("epoch"),
                "val_loss": ckpt.get("val_loss"),
                "val_auc": ckpt.get("val_auc"),
                "is_smoke_test": ckpt.get("is_smoke_test", True),
                "architecture": ckpt.get("architecture")
            }
        except Exception:
            pass

    history_file = settings.CHECKPOINTS_DIR / "training_history.json"
    history = None
    if history_file.exists():
        try:
            with open(history_file) as f:
                history = json.load(f)
        except Exception:
            pass

    return {
        "is_model_trained": has_weights,
        "is_smoke_test": ckpt_meta.get("is_smoke_test", True) if has_weights else None,
        "checkpoint_file": str(weights_path) if has_weights else None,
        "checkpoint_metadata": ckpt_meta,
        "training_history": history,
        "calibrator_fitted": predictor.calibrator.is_fitted if predictor.calibrator else False
    }

@router.get("/api/model/evaluation")
async def get_model_evaluation():
    """
    Returns actual scientific evaluation metrics and confusion matrix
    evaluated on held-out test events separated temporally and spatially.
    """
    eval_file = settings.CHECKPOINTS_DIR / "evaluation_results.json"
    if eval_file.exists():
        try:
            with open(eval_file) as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error reading evaluation results: {e}")
    
    return {
        "status": "pending_evaluation",
        "sample_count": 0,
        "metrics": None,
        "confusion_matrix": None,
        "evaluation_notes": ["Run evaluate.py to generate genuine held-out test evaluation."]
    }

@router.post("/api/predict", response_model=PredictResponse)
async def predict_flood_risk(req: PredictRequest):
    """
    Main 48-Hour Multimodal Flood Prediction Endpoint.
    1. Validates coordinates
    2. Fetches weather, satellite, and terrain data
    3. Runs multimodal inference or returns informative unavailable state
    """
    lat, lon = req.latitude, req.longitude
    logger.info(f"Received prediction request for ({lat}, {lon}) horizon={req.forecast_horizon_hours}h demo={req.simulation_demo_mode}")

    # 1. Fetch weather
    weather_dict = None
    try:
        w_resp = await weather_service.get_weather(lat=lat, lon=lon)
        weather_dict = w_resp.model_dump()
    except Exception as e:
        logger.warning(f"Could not retrieve weather for prediction ({lat}, {lon}): {e}")

    # 2. Fetch satellite
    sat_dict = None
    try:
        s_resp = await satellite_service.get_sentinel1_metadata(lat=lat, lon=lon)
        sat_dict = s_resp.model_dump()
    except Exception as e:
        logger.warning(f"Could not retrieve satellite info for prediction ({lat}, {lon}): {e}")

    # 3. Fetch terrain
    terrain_dict = None
    try:
        terrain_dict = await terrain_service.get_terrain_features(lat=lat, lon=lon)
    except Exception as e:
        logger.warning(f"Could not retrieve terrain for prediction ({lat}, {lon}): {e}")

    # 4. Predict
    prediction_response = predictor.predict(
        lat=lat,
        lon=lon,
        weather_data=weather_dict,
        sar_data=sat_dict,
        terrain_data=terrain_dict,
        forecast_horizon_hours=req.forecast_horizon_hours,
        demo_mode=req.simulation_demo_mode or False
    )

    return prediction_response
