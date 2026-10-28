# Dataset Preparation, Target Design & Provenance

## Target Label Definition

The primary machine learning forecasting target is defined as:
$$\text{flood\_within\_48h} \in \{0, 1\}$$

- **Positive Label ($\text{target} = 1$):** A documented inundation event begins or expands within the target geographic cell within the 48 hours immediately following the prediction timestamp $T_0$.
- **Negative Label ($\text{target} = 0$):** Verified absence of inundation during the 48-hour window, confirmed by multi-sensor optical/SAR observation and lack of disaster registry reports.
- **Uncertain / Cloud-Obscured Cases:** Flagged and excluded from training partitions to prevent noisy label corruption.

## Historical Ground Truth Catalogs

1. **Global Flood Database (GFD) - MODIS Events V1:**
   - Provider: Cloud to Street / Google Earth Engine
   - Catalog ID: `GLOBAL_FLOOD_DB_MODIS_EVENTS_V1`
   - Covers 913 large flood events globally from 2000 to 2018 at 250m resolution.
2. **Dartmouth Flood Observatory (DFO) Global Active Archive:**
   - Detailed event metadata including start date, end date, duration, flooded area ($\text{km}^2$), main cause, casualties, and displaced populations.

## Leakage Prevention Protocol

To ensure rigorous scientific validity:
1. **Event-Grouped Partitioning:** All geographic cells belonging to a specific historical flood event are strictly isolated into either Train, Validation, or Test splits. A model is never evaluated on cells from an event it has trained on.
2. **Temporal Reforecast Preservation:** Weather sequences prior to $T_0$ use historical reanalysis. Forecast features after $T_0$ use numerical weather prediction forecasts issued at or before $T_0$. Real observed future rain is never used as a proxy for forecasts during training.
3. **Modality Dropout:** During training, modalities (SAR patch or terrain) are randomly masked with 15% probability so the fusion network learns robust inference when a modality is temporarily unavailable.
