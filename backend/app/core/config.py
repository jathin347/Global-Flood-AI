import os
from pathlib import Path
from pydantic import BaseModel, Field

class Settings(BaseModel):
    PROJECT_NAME: str = "GlobalFlood AI"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api"
    DESCRIPTION: str = (
        "Global Multimodal Deep Learning System for Global Flood Forecasting "
        "up to 48 Hours Ahead integrating Sentinel-1 SAR, Open-Meteo, and Topographic Data."
    )
    
    # Environment
    ENVIRONMENT: str = Field(default_factory=lambda: os.getenv("ENVIRONMENT", "development"))
    DEBUG: bool = Field(default_factory=lambda: os.getenv("DEBUG", "true").lower() == "true")
    
    # Host & Port
    HOST: str = Field(default_factory=lambda: os.getenv("HOST", "0.0.0.0"))
    PORT: int = Field(default_factory=lambda: int(os.getenv("PORT", "8000")))
    
    # CORS
    CORS_ORIGINS: list[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "*"
    ]
    
    # External APIs
    OPEN_METEO_FORECAST_URL: str = "https://api.open-meteo.com/v1/forecast"
    OPEN_METEO_GEOCODING_URL: str = "https://geocoding-api.open-meteo.com/v1/search"
    OPEN_METEO_HISTORICAL_URL: str = "https://archive-api.open-meteo.com/v1/archive"
    OPEN_ELEVATION_URL: str = "https://api.open-elevation.com/api/v1/lookup"
    
    # Copernicus / Earth Engine (Optional configured tokens)
    COPERNICUS_CLIENT_ID: str = Field(default_factory=lambda: os.getenv("COPERNICUS_CLIENT_ID", ""))
    COPERNICUS_CLIENT_SECRET: str = Field(default_factory=lambda: os.getenv("COPERNICUS_CLIENT_SECRET", ""))
    EARTH_ENGINE_SERVICE_ACCOUNT: str = Field(default_factory=lambda: os.getenv("EARTH_ENGINE_SERVICE_ACCOUNT", ""))
    
    # Models & Data Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    MODELS_DIR: Path = BASE_DIR / "models"
    CHECKPOINTS_DIR: Path = MODELS_DIR / "checkpoints"
    MODEL_WEIGHTS_FILE: Path = CHECKPOINTS_DIR / "multimodal_flood_net_best.pt"
    CALIBRATION_FILE: Path = CHECKPOINTS_DIR / "calibrator.joblib"
    BASELINE_MODEL_FILE: Path = CHECKPOINTS_DIR / "baseline_xgboost.joblib"

    # Timeouts & Cache
    HTTP_TIMEOUT_SECONDS: float = 15.0
    CACHE_EXPIRY_SECONDS: int = 1800  # 30 minutes

settings = Settings()
