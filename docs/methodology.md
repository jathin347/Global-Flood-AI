# GlobalFlood AI: Scientific Methodology & Mathematical Formulation

## 1. Executive Summary & Physical Motivation

Flooding is a spatiotemporal hydro-meteorological process governed by the non-linear interaction of atmospheric forcing, antecedent catchment saturation, and localized surface topography:
- **Atmospheric Forcing:** Precipitation volume, intensity, and duration drive surface runoff.
- **Antecedent Soil Moisture:** A 50mm storm on parched, permeable soil may generate negligible runoff; the same storm on saturated soil (volumetric soil moisture $>0.35\text{ m}^3/\text{m}^3$) triggers rapid catastrophic overland flooding.
- **Topographic Concentration:** Water moves along gravitational hydraulic gradients toward low-lying depressions and natural drainage networks.
- **Microwave Radar Penetration:** Traditional optical remote sensing (Sentinel-2, Landsat) cannot penetrate the thick cloud decks accompanying flood-producing cyclones. Synthetic Aperture Radar (SAR) C-band microwaves (5.405 GHz) pass through cloud cover unimpeded, revealing surface water extent through specular reflectance.

GlobalFlood AI integrates these complementary modalities into a unified PyTorch neural network, **MultimodalFloodNet**, predicting the probability of flood onset within a 48-hour forward horizon:
$$\hat{p} = P\left(\text{flood within } 48\text{h} \mid \mathbf{X}_{\text{SAR}}, \mathbf{X}_{\text{Weather}}, \mathbf{X}_{\text{Terrain}}, \mathbf{m}\right)$$

---

## 2. Mathematical Formulation & Architecture

```
                  ┌──────────────────────────────────────────────┐
                  │ 1. Sentinel-1 SAR Dual-Pol (VV/VH 128x128)   │
                  └──────────────────────┬───────────────────────┘
                                         ▼
                     [ 4-Layer Conv2D + BatchNorm + GAP ]
                                         │  (128-d representation)
                                         ▼
┌────────────────────────────────────────┼────────────────────────────────────────┐
│ 2. Weather GRU (168h Obs + 48h Fc)     │ 3. Topographic MLP (Elevation, Slope) │
│ [ 2-Layer GRU + Missingness Masks ]    │ [ 2-Layer MLP + BatchNorm + Dropout ]  │
│             (64-d representation)      │             (32-d representation)      │
└───────────────────┬────────────────────┴───────────────────┬────────────────────┘
                    │                                        │
                    └────────────────────┬───────────────────┘
                                         ▼
                 ┌──────────────────────────────────────────────┐
                 │      Mask-Aware Fusion Layer (227-d)         │
                 │ Dense(128) + Dropout(0.3) + Dense(64) + Logit│
                 └───────────────────────┬──────────────────────┘
                                         ▼
                 ┌──────────────────────────────────────────────┐
                 │     Platt Scaling Probability Calibration    │
                 │   Low (<25%) | Mod (25-55%) | High | V.High  │
                 └──────────────────────────────────────────────┘
```

### Branch 1: Synthetic Aperture Radar (SAR) CNN Encoder
- **Input Tensor:** $\mathbf{X}_{\text{SAR}} \in \mathbb{R}^{B \times 2 \times 128 \times 128}$ representing calibrated backscatter intensity $\sigma^0$ in VV and VH polarizations.
- **Physics of SAR Water Detection:** Smooth water surfaces act as specular reflectors, scattering incoming microwave radar pulses away from the receiver antenna. Consequently, water bodies appear with distinct low radar backscatter ($-25\text{ dB}$ to $-18\text{ dB}$ in VV/VH), whereas terrain, vegetation, and buildings produce rough diffuse backscatter ($-14\text{ dB}$ to $-6\text{ dB}$).
- **Normalization:** Raw values in decibels are clipped to $[-30.0, 0.0]\text{ dB}$ and mapped to $[0.0, 1.0]$:
  $$\mathbf{X}_{\text{norm}} = \frac{\text{clip}(\mathbf{X}_{\text{dB}}, -30.0, 0.0) + 30.0}{30.0}$$
- **Encoder Structure:**
  - Stage 1: $\text{Conv2D}(2 \to 32, 3\times 3, \text{pad } 1) \to \text{BatchNorm} \to \text{ReLU} \to \text{MaxPool}(2\times 2) \implies (64\times 64)$
  - Stage 2: $\text{Conv2D}(32 \to 64, 3\times 3, \text{pad } 1) \to \text{BatchNorm} \to \text{ReLU} \to \text{MaxPool}(2\times 2) \implies (32\times 32)$
  - Stage 3: $\text{Conv2D}(64 \to 128, 3\times 3, \text{pad } 1) \to \text{BatchNorm} \to \text{ReLU} \to \text{MaxPool}(2\times 2) \implies (16\times 16)$
  - Stage 4: $\text{Conv2D}(128 \to 128, 3\times 3, \text{pad } 1) \to \text{BatchNorm} \to \text{ReLU} \to \text{AdaptiveAvgPool2D}(1\times 1) \implies (128)$
  - Projection: $\text{Dense}(128 \to 128) \to \text{LayerNorm} \to \text{ReLU} \implies \mathbf{h}_{\text{SAR}} \in \mathbb{R}^{128}$

### Branch 2: Weather Sequence GRU Encoder
- **Historical Sequence:** 168 hours (7 days) of antecedent reanalysis observations $\mathbf{S}_{\text{hist}} \in \mathbb{R}^{B \times 168 \times 6}$ with binary mask $\mathbf{M}_{\text{hist}} \in \{0, 1\}^{B \times 168 \times 6}$.
- **Forecast Sequence:** 48 hours of numerical weather prediction $\mathbf{S}_{\text{fc}} \in \mathbb{R}^{B \times 48 \times 6}$ with binary mask $\mathbf{M}_{\text{fc}} \in \{0, 1\}^{B \times 48 \times 6}$.
- **Variables ($F=6$):**
  1. Precipitation (mm)
  2. 2m Temperature (°C)
  3. 2m Relative Humidity (%)
  4. Surface Pressure (hPa)
  5. 10m Wind Speed (km/h)
  6. Topsoil Moisture ($0\text{--}1\text{cm}, \text{m}^3/\text{m}^3$)
- **Missing-Value Mechanism:** Values and masks are concatenated along the feature dimension ($\text{dim}=12$), ensuring that missing values (e.g. soil moisture in future forecasts) are explicitly encoded rather than arbitrarily imputed:
  $$\mathbf{h}_{\text{hist}} = \text{GRU}_{\text{2-layer}}\left([\mathbf{S}_{\text{hist}} \,\|\, \mathbf{M}_{\text{hist}}]\right) \in \mathbb{R}^{64}$$
  $$\mathbf{h}_{\text{fc}} = \text{GRU}_{\text{1-layer}}\left([\mathbf{S}_{\text{fc}} \,\|\, \mathbf{M}_{\text{fc}}]\right) \in \mathbb{R}^{64}$$
  $$\mathbf{h}_{\text{weather}} = \text{Dense}(128 \to 64) \left([\mathbf{h}_{\text{hist}} \,\|\, \mathbf{h}_{\text{fc}}]\right) \in \mathbb{R}^{64}$$

### Branch 3: Topographic Terrain MLP Encoder
- **Input Features ($D=9$):**
  1. Elevation ($m$ ASL) from Copernicus DEM / Open-Elevation
  2. Topographic slope gradient ($\beta$ in degrees)
  3. Terrain Roughness Index (TRI)
  4. Topographic Wetness Index proxy ($\text{TWI} = \ln(a / \tan \beta)$)
  5. Distance to nearest drainage / coastline proxy ($km$)
  6. Spherical coordinate cyclic encodings: $[\sin(\text{lat}), \cos(\text{lat}), \sin(\text{lon}), \cos(\text{lon})]$
- **Encoder Structure:**
  $$\mathbf{h}_{\text{geo}} = \text{Dense}(9 \to 64) \to \text{BatchNorm} \to \text{ReLU} \to \text{Dropout}(0.2) \to \text{Dense}(64 \to 32) \implies \mathbf{h}_{\text{geo}} \in \mathbb{R}^{32}$$

### Branch 4: Mask-Aware Fusion Layer
- The three representation vectors contribute $128 + 64 + 32 = 224$ dimensions.
- Three binary modality availability indicators $\mathbf{m} = [m_{\text{SAR}}, m_{\text{weather}}, m_{\text{geo}}] \in \{0, 1\}^3$ are concatenated:
  $$\mathbf{z}_{\text{fusion}} = [\mathbf{h}_{\text{SAR}} \cdot m_{\text{SAR}} \,\|\, \mathbf{h}_{\text{weather}} \cdot m_{\text{weather}} \,\|\, \mathbf{h}_{\text{geo}} \cdot m_{\text{geo}} \,\|\, \mathbf{m}] \in \mathbb{R}^{227}$$
- **Fusion Multi-Layer Perceptron:**
  $$\mathbf{u}_1 = \text{ReLU}(\mathbf{W}_1 \mathbf{z}_{\text{fusion}} + \mathbf{b}_1), \quad \mathbf{W}_1 \in \mathbb{R}^{128 \times 227}$$
  $$\mathbf{u}_2 = \text{Dropout}_{0.3}(\mathbf{u}_1)$$
  $$\mathbf{u}_3 = \text{ReLU}(\mathbf{W}_2 \mathbf{u}_2 + \mathbf{b}_2), \quad \mathbf{W}_2 \in \mathbb{R}^{64 \times 128}$$
  $$\hat{z} = \mathbf{w}_3^T \mathbf{u}_3 + b_3, \quad \hat{z} \in \mathbb{R} \quad (\text{Logit})$$
  $$p_{\text{raw}} = \sigma(\hat{z}) = \frac{1}{1 + \exp(-\hat{z})}$$

---

## 3. Data Leakage Prevention Protocol

Data leakage is the single greatest cause of failure in operational academic machine learning for geoscience. GlobalFlood AI enforces three strict safeguards:

1. **Strict Temporal Boundary Enforcement:**
   - Observations before $T_0$ use reanalysis / observations.
   - Predictions after $T_0$ use numerical weather prediction forecasts.
   - Future observed rainfall after $T_0$ is **never** used during training as a proxy for forecast rainfall.

2. **Event-Grouped Cross-Validation (Spatial & Temporal Disjointness):**
   - Samples are grouped strictly by **Event ID**.
   - If Event A occurred in Chennai in Nov 2015, all spatial points within that basin during that event are assigned to either Train or Test, never split across both.
   - Test events (**GFD_MODIS_4210 Sylhet Basin** and **GFD_MODIS_3350 Yorkshire River**) were evaluated exclusively after all training and parameter tuning had completed.

3. **Modality Dropout:**
   - During training, modality presence flags $m_i$ are randomly dropped out with $p=0.15$, teaching the fusion layer to produce valid inferences even when Sentinel-1 lacks a recent pass or terrain lookups are delayed.

---

## 4. Loss Function & Class Imbalance

Floods represent rare, high-consequence phenomena (positive rate $<10\%$). Unweighted cross-entropy causes models to degenerate into trivial majority-class classifiers. We utilize Positive-Class Weighted Binary Cross-Entropy with Logits:

$$\mathcal{L}(\mathbf{y}, \hat{\mathbf{z}}) = -\frac{1}{B} \sum_{i=1}^B \left[ w_+ \cdot y_i \log \sigma(\hat{z}_i) + (1 - y_i) \log(1 - \sigma(\hat{z}_i)) \right]$$

where the positive weight is dynamically set to:
$$w_+ = \frac{N_{\text{negative}}}{N_{\text{positive}}}$$

Optimization is performed using AdamW with weight decay $\lambda = 10^{-4}$, gradient clipping at $\|\mathbf{g}\|_2 \le 1.0$, and learning rate scheduling via `ReduceLROnPlateau(factor=0.5, patience=3)`.

---

## 5. Probability Calibration via Platt Scaling

Deep neural networks trained with cross-entropy frequently suffer from miscalibration: predicted probabilities $\sigma(\hat{z})$ tend to cluster near 0 and 1, exaggerating confidence.

To ensure that a predicted probability of $70\%$ corresponds empirically to a $70\%$ frequency of flood occurrences, we fit a univariate logistic regression (Platt scaling) on the validation set logits:
$$P(\text{flood} \mid \hat{z}) = \frac{1}{1 + \exp(A \cdot \hat{z} + B)}$$

- Calibration parameters $A$ and $B$ are estimated exclusively on the held-out validation partition.
- Calibration is completely isolated from training loss backpropagation.
- The fitted calibrator is serialized to `models/checkpoints/calibrator.joblib` and applied during live inference.

---

## 6. Operational Risk Categories

Calibrated probabilities are mapped to intuitive application categories:

| Risk Category | Calibrated Probability Interval | Interpretation & Guidance |
| :--- | :---: | :--- |
| **Low** | $[0.00, 0.25)$ | Baseline hydrological safety; normal conditions. |
| **Moderate** | $[0.25, 0.55)$ | Elevated precipitation or low slope; localized ponding possible. |
| **High** | $[0.55, 0.80)$ | Heavy rainfall combined with saturated antecedent soil or low-lying basin. |
| **Very High** | $[0.80, 1.00]$ | Severe multimodal convergence; extreme precipitation on saturated lowland. |

> **IMPORTANT DISCLAIMER:** These risk tiers are experimental application categories for research exploration. They DO NOT constitute official civil defense warnings or evacuation notices. Always refer to authorized national hydro-meteorological agencies.

---

## 7. Evaluation Metrics & Scientific Criteria

Evaluation is performed using established scientific criteria on the held-out test events:
- **ROC-AUC (Receiver Operating Characteristic Area Under Curve):** Measures discrimination ability across all classification thresholds.
- **PR-AUC (Precision-Recall Area Under Curve):** Measures performance on imbalanced positive classes.
- **F1-Score ($2 \cdot \frac{P \cdot R}{P + R}$):** Harmonic mean of precision and recall at default decision threshold $0.50$.
- **Brier Score ($\frac{1}{N} \sum (\hat{p}_i - y_i)^2$):** Measures mean squared probability error; lower Brier scores indicate superior calibration reliability.
- **Confusion Matrix:** Explicit count of True Negatives (TN), False Positives (FP), False Negatives (FN), and True Positives (TP).
