# GlobalFlood AI Architecture

## System Overview

GlobalFlood AI is a multimodal deep learning system designed for 48-hour global flood forecasting. The system integrates three heterogeneous spatial and temporal data streams:
1. **Copernicus Sentinel-1 SAR** (Synthetic Aperture Radar) backscatter imagery (VV and VH polarizations) at 10m spatial resolution.
2. **Open-Meteo High-Resolution Numerical Weather Predictions** (168h historical observations + 48h future forecast).
3. **Copernicus DEM (90m)** providing elevation, slope, terrain roughness (TRI), and topographic wetness (TWI).

```mermaid
graph TD
    subgraph Data_Acquisition ["Modality Ingestion Layer"]
        S1["Copernicus Sentinel-1 C-SAR<br/>(VV/VH Polarizations, 128x128 Patch)"]
        OM["Open-Meteo Weather API<br/>(168h Historical Obs + 48h Numerical Forecast)"]
        DEM["Copernicus DEM 90m<br/>(Elevation, Slope, TRI, TWI, River Dist)"]
    end

    subgraph Preprocessing_Layer ["Leakage-Free Preprocessing"]
        P_SAR["Decibel Conversion & Normalization<br/>[-30 dB, 0 dB] &rarr; [0.0, 1.0]"]
        P_WTH["Z-Score Standardization & Missingness Masking<br/>(Separated 168h Obs vs 48h Forecast)"]
        P_GEO["Topographic Standardization & Spherical Coordinate Encoding<br/>[sin/cos lat, sin/cos lon]"]
    end

    S1 --> P_SAR
    OM --> P_WTH
    DEM --> P_GEO

    subgraph Multimodal_Neural_Network ["Multimodal Deep Learning Architecture"]
        subgraph Branch1 ["Branch 1: Satellite Image CNN Encoder"]
            B1_Conv["4x Conv2D + BatchNorm + ReLU + MaxPool"]
            B1_Pool["Global Average Pooling"]
            B1_Proj["FC Projection &rarr; 128 Dimensions"]
            B1_Conv --> B1_Pool --> B1_Proj
        end

        subgraph Branch2 ["Branch 2: Weather Time-Series GRU"]
            B2_GRU1["2-Layer Bidirectional GRU (168h Obs + Mask)"]
            B2_GRU2["1-Layer Forecast GRU (48h Forecast + Mask)"]
            B2_Proj["Combined Temporal Projection &rarr; 64 Dimensions"]
            B2_GRU1 --> B2_Proj
            B2_GRU2 --> B2_Proj
        end

        subgraph Branch3 ["Branch 3: Geographic Feature MLP"]
            B3_MLP["2-Layer MLP + BatchNorm + Dropout(0.2)"]
            B3_Proj["Linear Projection &rarr; 32 Dimensions"]
            B3_MLP --> B3_Proj
        end

        subgraph Fusion ["Mask-Aware Fusion Network"]
            Concat["Feature Concatenation + Modality Availability Flags<br/>128d + 64d + 32d + 3d = 227 Dimensions"]
            Dense1["Dense(128, ReLU) + Dropout(0.3)"]
            Dense2["Dense(64, ReLU) + Dropout(0.2)"]
            Logit["Output Layer (1 Logit)"]
            Concat --> Dense1 --> Dense2 --> Logit
        end
    end

    P_SAR --> B1_Conv
    P_WTH --> B2_GRU1
    P_WTH --> B2_GRU2
    P_GEO --> B3_MLP

    B1_Proj --> Concat
    B2_Proj --> Concat
    B3_Proj --> Concat

    subgraph Calibration_Inference ["Calibration & Risk Engine"]
        Platt["Platt Scaling / Isotonic Calibrator"]
        RiskClass["Calibrated Threshold Classifier<br/>Low (<25%) | Moderate (25-55%) | High (55-80%) | Very High (>80%)"]
        Uncertainty["Epistemic & Aleatoric Uncertainty Bounds (90% CI)"]
        Logit --> Platt --> RiskClass
        Logit --> Uncertainty
    end

    subgraph Presentation_Layer ["Presentation & API Layer"]
        FastAPI["FastAPI Backend Service<br/>(/api/predict, /api/weather, /api/satellite)"]
        ReactUI["React + Vite + Leaflet Scientific Dashboard<br/>(Risk Footprints, Historical Overlays, Recharts)"]
        RiskClass --> FastAPI
        Uncertainty --> FastAPI
        FastAPI --> ReactUI
    end
```

## Modality Specifications

| Modality | Sensor / Source | Spatial Resolution | Temporal Frequency | Representation / Output |
| :--- | :--- | :--- | :--- | :--- |
| **SAR Imagery** | Sentinel-1 C-SAR (IW GRD) | 10 meters | 6-12 day repeat orbit | 128x128 patch, 2 channels (VV, VH) &rarr; 128-d embedding |
| **Historical Weather** | Open-Meteo Reanalysis | ~11 km (0.1°) | Hourly (168 hours) | 6 variables (rain, temp, humidity, pressure, wind, soil) + mask &rarr; 64-d GRU |
| **Forecast Weather** | Open-Meteo GFS/ECMWF | ~11 km (0.1°) | Hourly (48 hours) | 6 variables (forecast rain, temp, humidity, pressure, wind) + mask &rarr; 64-d GRU |
| **Terrain Topography** | Copernicus DEM / Open-Elevation | 90 meters | Static WGS84 | Elevation, Slope, TRI, TWI, River Distance, Sin/Cos Lat/Lon &rarr; 32-d MLP |
