# Model Limitations, Uncertainty & Operational Disclaimers

## Technical and Geospatial Limitations

1. **Satellite Orbital Repeat Cycle Latency:**
   Sentinel-1 SAR operates on a 6-to-12 day orbital repeat cycle. In real-time operational deployment, the most recent available radar scene may be 2 to 5 days old, capturing antecedent surface moisture rather than instant flood onset.
2. **Spatial Scale Discrepancies:**
   While Sentinel-1 provides 10m spatial resolution and Copernicus DEM provides 90m resolution, numerical weather prediction (NWP) grids (ECMWF IFS / GFS via Open-Meteo) operate at 9km to 25km grid resolution. Localized convective cloudbursts ("microbursts") may fall between grid points.
3. **Sub-Surface Drainage & Infrastructure:**
   The model does not model subterranean urban drainage pipe networks, pumping station failures, or localized culvert blockages, which frequently drive urban flash flooding independently of regional slope.
4. **Proxy Ground Truth Uncertainty:**
   Satellite-derived historical flood extents (e.g., MODIS 250m) have mixed-pixel uncertainties and may miss narrow stream overflows beneath dense canopy.

## Operational & Ethical Disclaimer

> [!WARNING]
> **RESEARCH AND EDUCATIONAL PURPOSES ONLY**
> GlobalFlood AI is an experimental academic prototype developed for research in multimodal geospatial artificial intelligence.
> Predictions produced by this system represent probabilistic deep learning inferences and **MUST NOT** be used as official flood alerts, life-safety guidance, or civil defense evacuation orders.
> Always consult official bulletins from your national hydro-meteorological agency and local emergency management authorities.
