import math
import numpy as np
from typing import Optional

class MultimodalPreprocessor:
    """
    Standardizes and normalizes features across all 3 modalities:
    1. Sentinel-1 SAR: dB backscatter normalization [-30 dB, 0 dB] -> [0.0, 1.0]
    2. Weather sequences: 6 features [precip, temp, humidity, pressure, wind, soil]
    3. Geographic features: [elev, slope, tri, twi, dist_river, sin_lat, cos_lat, sin_lon, cos_lon]
    """
    # Reference mean and std statistics calculated from global hydrological climatology
    WEATHER_MEANS = np.array([0.5, 18.0, 70.0, 1013.25, 12.0, 0.28], dtype=np.float32)
    WEATHER_STDS = np.array([3.5, 12.0, 20.0, 15.0, 8.0, 0.12], dtype=np.float32)

    GEO_MEANS = np.array([120.0, 4.5, 8.0, 7.5, 15.0], dtype=np.float32)  # elev, slope, tri, twi, dist_river
    GEO_STDS = np.array([250.0, 6.0, 10.0, 3.0, 20.0], dtype=np.float32)

    @classmethod
    def preprocess_sar_patch(cls, patch: Optional[np.ndarray]) -> tuple[np.ndarray, float]:
        """
        patch: np.ndarray shape (2, 128, 128) with VV and VH channels
        Returns: (normalized_patch, availability_flag)
        """
        if patch is None or patch.size == 0:
            # Missing SAR modality -> zero array and mask = 0.0
            return np.zeros((2, 128, 128), dtype=np.float32), 0.0

        # Ensure correct shape
        if patch.shape != (2, 128, 128):
            # Reshape or pad if needed
            new_patch = np.zeros((2, 128, 128), dtype=np.float32)
            c = min(2, patch.shape[0])
            h = min(128, patch.shape[1])
            w = min(128, patch.shape[2])
            new_patch[:c, :h, :w] = patch[:c, :h, :w]
            patch = new_patch

        # Clip dB between -30 dB and 0 dB, normalize to [0, 1]
        clipped = np.clip(patch, -30.0, 0.0)
        normalized = (clipped + 30.0) / 30.0
        return normalized.astype(np.float32), 1.0

    @classmethod
    def preprocess_weather_sequence(
        cls,
        hourly_records: list[dict],
        target_hours: int = 168
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Transforms raw hourly records into normalized arrays and missingness masks.
        Returns:
          values: (target_hours, 6)
          mask: (target_hours, 6) where 1.0 means observed, 0.0 means missing
        """
        values = np.zeros((target_hours, 6), dtype=np.float32)
        mask = np.zeros((target_hours, 6), dtype=np.float32)

        keys = [
            "precipitation_mm",
            "temperature_2m_c",
            "relative_humidity_pct",
            "surface_pressure_hpa",
            "wind_speed_10m_kmh",
            "soil_moisture_m3m3"
        ]

        # Fill available records (right-aligned to end at prediction time)
        num_records = min(len(hourly_records), target_hours)
        offset = target_hours - num_records

        for i in range(num_records):
            rec = hourly_records[i]
            for j, key in enumerate(keys):
                val = rec.get(key)
                if val is not None and not math.isnan(val):
                    # Normalized value
                    norm_val = (val - cls.WEATHER_MEANS[j]) / (cls.WEATHER_STDS[j] + 1e-6)
                    values[offset + i, j] = norm_val
                    mask[offset + i, j] = 1.0
                else:
                    values[offset + i, j] = 0.0
                    mask[offset + i, j] = 0.0

        return values, mask

    @classmethod
    def preprocess_geo_features(
        cls,
        lat: float,
        lon: float,
        terrain_dict: dict
    ) -> tuple[np.ndarray, float]:
        """
        Extracts 9-dimensional geographic vector:
        [elev_norm, slope_norm, tri_norm, twi_norm, dist_river_norm, sin_lat, cos_lat, sin_lon, cos_lon]
        """
        elev = terrain_dict.get("elevation_m", 15.0)
        slope = terrain_dict.get("slope_degrees", 2.0)
        tri = terrain_dict.get("terrain_roughness_index", 5.0)
        twi = terrain_dict.get("topographic_wetness_index", 7.0)
        dist_river = terrain_dict.get("estimated_dist_to_drainage_km", 5.0)

        raw_geo = np.array([elev, slope, tri, twi, dist_river], dtype=np.float32)
        norm_geo = (raw_geo - cls.GEO_MEANS) / (cls.GEO_STDS + 1e-6)

        # Cyclic encoding for latitude and longitude (WGS84 spherical coordinates)
        lat_rad = math.radians(lat)
        lon_rad = math.radians(lon)
        sin_lat = math.sin(lat_rad)
        cos_lat = math.cos(lat_rad)
        sin_lon = math.sin(lon_rad)
        cos_lon = math.cos(lon_rad)

        full_vec = np.concatenate([norm_geo, [sin_lat, cos_lat, sin_lon, cos_lon]]).astype(np.float32)
        return full_vec, 1.0
