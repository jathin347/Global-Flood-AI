from typing import Optional, Literal
from pydantic import BaseModel, Field

class PredictRequest(BaseModel):
    latitude: float = Field(..., ge=-90.0, le=90.0, description="Latitude between -90 and 90")
    longitude: float = Field(..., ge=-180.0, le=180.0, description="Longitude between -180 and 180")
    forecast_horizon_hours: int = Field(default=48, ge=6, le=168, description="Prediction horizon (hours ahead)")
    region_size_km: Optional[float] = Field(default=10.0, ge=1.0, le=100.0, description="Local analysis radius in km")
    simulation_demo_mode: Optional[bool] = Field(default=False, description="Explicit user request for demo/simulation walk-through")

class DataSourceStatus(BaseModel):
    name: str
    modality: str  # "weather_forecast", "weather_historical", "satellite_sar", "geographic_terrain"
    status: Literal["available", "partial", "unavailable", "mock_demo"]
    retrieval_timestamp: Optional[str] = None
    latency_ms: Optional[float] = None
    source_attribution: str

class FactorImportance(BaseModel):
    factor_name: str
    feature_group: str  # "weather", "terrain", "satellite"
    value: float
    unit: str
    impact: Literal["increasing_risk", "neutral", "decreasing_risk"]
    relative_weight: float  # 0.0 to 1.0

class UncertaintyInfo(BaseModel):
    confidence_score: float  # 0.0 to 1.0
    epistemic_uncertainty: float
    aleatoric_uncertainty: float
    lower_bound_90ci: float
    upper_bound_90ci: float

class PredictResponse(BaseModel):
    status: Literal["success", "partial_modalities", "unavailable", "model_not_trained", "demo"]
    location: dict
    forecast_horizon_hours: int = 48
    flood_probability: Optional[float] = None
    risk_category: Literal["low", "moderate", "high", "very_high", "unavailable"]
    model_version: str
    model_architecture: str
    data_sources: list[DataSourceStatus]
    missing_modalities: list[str]
    uncertainty: Optional[UncertaintyInfo] = None
    factor_importances: list[FactorImportance] = Field(default_factory=list)
    calibrated_thresholds: dict = Field(
        default_factory=lambda: {
            "low": [0.0, 0.25],
            "moderate": [0.25, 0.55],
            "high": [0.55, 0.80],
            "very_high": [0.80, 1.00]
        }
    )
    prediction_timestamp: str
    disclaimer: str = (
        "RESEARCH DISCLAIMER: GlobalFlood AI predictions are experimental deep learning forecasts "
        "for academic and scientific evaluation. They DO NOT constitute official flood warnings, emergency alerts, "
        "or life-safety guidance. Always refer to official national hydro-meteorological services during flood threats."
    )
    limitations: list[str] = Field(default_factory=list)
