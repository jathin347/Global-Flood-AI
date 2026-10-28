from typing import Optional
from pydantic import BaseModel, Field

class LocationSearchResult(BaseModel):
    id: Optional[int] = None
    name: str
    latitude: float
    longitude: float
    country: Optional[str] = None
    country_code: Optional[str] = None
    admin1: Optional[str] = None
    elevation: Optional[float] = None
    timezone: Optional[str] = None

class LocationSearchResponse(BaseModel):
    query: str
    count: int
    results: list[LocationSearchResult]

class CoordinatesQuery(BaseModel):
    latitude: float = Field(..., ge=-90.0, le=90.0, description="Latitude between -90 and 90")
    longitude: float = Field(..., ge=-180.0, le=180.0, description="Longitude between -180 and 180")
