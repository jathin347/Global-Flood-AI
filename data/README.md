# GlobalFlood AI Data Directory

This directory stores raw geospatial datasets, reanalysis cache, and processed aligned training samples.

## Structure
- `raw/`: Raw GeoTIFFs, Sentinel-1 zip products, GFD shapefiles, and historical weather JSON dumps.
- `processed/`: Aligned multimodal sample datasets formatted as JSON or Parquet with `(sar_patch, hist_weather_168h, fc_weather_48h, terrain_features, target_flood_within_48h)`.

## Data Sources & Acquisition
1. **Weather:** Automatically fetched and cached via `Open-Meteo API` (`https://api.open-meteo.com`).
2. **Satellite SAR:** Copernicus Sentinel-1 C-SAR IW GRD accessed via `Copernicus Data Space Ecosystem (CDSE)` (`https://dataspace.copernicus.eu`).
3. **Elevation:** Copernicus DEM 90m and Open-Elevation (`https://api.open-elevation.com`).
4. **Historical Flood Ground Truth:** Global Flood Database MODIS Events V1 (`https://developers.google.com/earth-engine/datasets/catalog/GLOBAL_FLOOD_DB_MODIS_EVENTS_V1`).
