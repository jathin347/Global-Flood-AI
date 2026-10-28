from typing import Optional
from pydantic import BaseModel

class HistoricalFloodEvent(BaseModel):
    event_id: str
    country: str
    region_name: str
    latitude: float
    longitude: float
    start_date: str
    end_date: str
    duration_days: int
    flooded_area_sq_km: Optional[float] = None
    severity_class: int  # 1: Large, 2: Very Large, 3: Extreme
    main_cause: str
    dead: Optional[int] = None
    displaced: Optional[int] = None
    source_database: str  # "Global Flood DB (MODIS Events V1)" or "DFO Archive"
    distance_km: Optional[float] = None
    historical_flood_within_48h_label: int = 1

class FloodHistoryResponse(BaseModel):
    status: str = "success"
    query_location: dict
    radius_km: float
    total_events_found: int
    events: list[HistoricalFloodEvent]
    database_version: str = "Global Flood DB v1.0 & DFO Global Active Archive"
    disclaimer: str = (
        "Historical flood extents document past inundated surfaces. Mapped flood extent does not guarantee "
        "future flood occurrence without active precipitation, soil saturation, and hydrological triggers."
    )
