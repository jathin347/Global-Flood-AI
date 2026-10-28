from typing import Optional
from pydantic import BaseModel, Field

class SatellitePatchMetadata(BaseModel):
    satellite: str = "Sentinel-1 SAR"
    platform: str = "Sentinel-1A/B"
    instrument: str = "C-SAR"
    mode: str = "IW"  # Interferometric Wide Swath
    polarizations: list[str] = ["VV", "VH"]
    product_type: str = "GRD"  # Ground Range Detected
    resolution_m: int = 10
    patch_size_pixels: int = 128
    orbit_direction: Optional[str] = None  # ASCENDING / DESCENDING
    relative_orbit: Optional[int] = None
    acquisition_time: Optional[str] = None
    scene_id: Optional[str] = None

class SatelliteResponse(BaseModel):
    status: str  # "available", "coverage_confirmed", "unavailable", "authentication_required"
    location: dict
    data_source: str = "Copernicus Data Space Ecosystem (Sentinel-1 SAR)"
    metadata: Optional[SatellitePatchMetadata] = None
    mean_backscatter_vv_db: Optional[float] = None
    mean_backscatter_vh_db: Optional[float] = None
    water_index_proxy: Optional[float] = None
    preview_url: Optional[str] = None
    footprint_geojson: Optional[dict] = None
    access_requirements: str = (
        "Free global Copernicus credentials (CDSE client credentials) required for full binary GRD swath download. "
        "Public spatial footprint & acquisition catalog accessed."
    )
    disclaimer: str = (
        "Satellite SAR provides cloud-penetrating water detection. Never substitute synthetic or unverified imagery as real data."
    )
