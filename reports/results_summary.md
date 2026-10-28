# Results Summary & Discussion: GlobalFlood AI (48-Hour Forecast)

This document provides a consolidated scientific summary of all recorded training, evaluation, and calibration metrics for the **GlobalFlood AI** system, accompanied by an evidence-based academic discussion for the project report.

---

## 1. Consolidated Evaluation Benchmark Table

All values below reflect exact, unrounded numbers transcribed directly from [`models/checkpoints/full_evaluation_metrics.json`](file:///Users/jathinaraveti/Desktop/ML/GlobalFloodAI/models/checkpoints/full_evaluation_metrics.json) on the held-out test split ($N=8$ samples across Bangladesh and UK events).

| Evaluation Metric | MultimodalFloodNet (Raw Uncalibrated) | MultimodalFloodNet (Platt Calibrated) | Tabular Logistic Baseline (Rain + Terrain) | Prior Frequency Baseline (Trivial) | Desired Direction |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **ROC-AUC** | **1.0000** | **1.0000** | 0.9375 | 0.5000 | Higher $\uparrow$ |
| **PR-AUC (Average Precision)** | **1.0000** | **1.0000** | 0.9437 | 0.7500 | Higher $\uparrow$ |
| **Accuracy (Threshold 0.50)** | **1.0000** (8/8) | **1.0000** (8/8) | 0.8750 (7/8) | 0.5000 (4/8) | Higher $\uparrow$ |
| **Precision** | **1.0000** | **1.0000** | 0.8000 | 0.5000 | Higher $\uparrow$ |
| **Recall (Sensitivity)** | **1.0000** | **1.0000** | 1.0000 | 1.0000 | Higher $\uparrow$ |
| **F1-Score** | **1.0000** | **1.0000** | 0.8889 | 0.6667 | Higher $\uparrow$ |
| **Specificity (True Negative Rate)** | **1.0000** | **1.0000** | 0.7500 | 0.0000 | Higher $\uparrow$ |
| **Balanced Accuracy** | **1.0000** | **1.0000** | 0.8750 | 0.5000 | Higher $\uparrow$ |
| **Optimal Threshold** | 0.9797 | 0.6543 | 0.5330 | 0.5000 | — |
| **Optimal F1-Score** | 1.0000 | 1.0000 | 0.8889 | 0.6667 | Higher $\uparrow$ |
| **Brier Score** | **0.000400** | 0.119559 | 0.090756 | 0.250000 | **Lower** $\downarrow$ |
| **Log Loss (Binary Cross-Entropy)** | **0.0201** | 0.4243 | 0.2582 | 0.6931 | **Lower** $\downarrow$ |
| **Expected Calibration Error (ECE)** | **0.0199** | 0.3458 | 0.0967 | 0.0000 | **Lower** $\downarrow$ |
| **Maximum Calibration Error (MCE)** | **0.0203** | 0.3461 | 0.4670 | 0.0000 | **Lower** $\downarrow$ |
| **Confusion Matrix [TN, FP; FN, TP]** | `[4, 0; 0, 4]` | `[4, 0; 0, 4]` | `[3, 1; 0, 4]` | `[0, 4; 0, 4]` | — |

---

## 2. Training Convergence History

*Source:* [`models/checkpoints/training_history.json`](file:///Users/jathinaraveti/Desktop/ML/GlobalFloodAI/models/checkpoints/training_history.json)  
*Optimization Details:* AdamW, Initial LR = $10^{-3}$, Weight Decay = $10^{-4}$, Batch Size = 8, Loss = Class-Weighted BCE.

| Epoch | Training Loss (BCE) | Validation Loss (BCE) | Validation ROC-AUC | Validation F1-Score |
| :---: | :---: | :---: | :---: | :---: |
| 1 | 0.600857 | 0.651023 | 1.0000 | 0.0000 |
| 2 | 0.396306 | 0.446056 | 1.0000 | 1.0000 |
| 3 | 0.234581 | 0.135900 | 1.0000 | 1.0000 |
| 4 | 0.361859 | 0.051902 | 1.0000 | 1.0000 |
| 5 | **0.038329** | **0.019816** | **1.0000** | **1.0000** |

*Note:* Training and validation accuracies were not recorded during training. Best checkpoint was saved at Epoch 5 (`val_loss = 0.0198`).

---

## 3. Evidence-Based Academic Discussion

### 3.1 Methodological Strengths
1. **Zero Spatial-Temporal Cross-Contamination:** Splitting is performed strictly by flood event (Event ID) rather than random sample shuffling. Events `GFD_MODIS_4210` (Sylhet Basin, Bangladesh) and `GFD_MODIS_3350` (Yorkshire River, UK) were kept entirely in the held-out test split, ensuring the model never trained on spatial coordinates or temporal windows from these basins.
2. **Standardization Protocol:** Normalization statistics for weather and terrain were defined from fixed hydrological climatology constants rather than computed across the dataset, eliminating preprocessing leakage from the test split.
3. **Multimodal Late Fusion:** The architecture effectively combines 2D satellite spatial embeddings (128d), temporal sequential dynamics via GRU (64d), and topographic characteristics via MLP (32d), outperforming the tabular baseline (accuracy $1.000$ vs $0.875$) and trivial baseline ($0.500$).

### 3.2 Analysis of the Platt Calibration Paradox
A critical finding from the evaluation is that **Platt scaling significantly degraded probabilistic reliability**:
- **Raw Network Output:** Brier score = $0.0004$, $\text{ECE} = 0.0199$.
- **Platt Calibrated Output:** Brier score = $0.1196$, $\text{ECE} = 0.3458$.

**Mechanism:**  
In the raw network, positive test samples produced uncalibrated sigmoid probabilities of $\approx 0.980$, and negative samples produced $\approx 0.020$. Because the test labels are binary ($1$ and $0$), these extreme predictions yielded near-zero squared probability error ($\text{Brier} = 0.0004$).  
When univariate logistic regression (Platt scaling) fitted on the small validation partition ($N=8$) was applied, it pulled extreme probabilities inward toward the validation mean, assigning flooded samples $\approx 0.655$ and non-flooded samples $\approx 0.346$. This contraction introduced large calibration gaps ($\approx 0.346$) in both populated bins, worsening the Brier score by a factor of nearly 300.  
**Report Recommendation:** Document this explicitly as an important methodological finding: post-hoc parametric probability calibration is highly volatile when fitted or evaluated on small-sample datasets.

### 3.3 Core Scientific Limitations & Realities
To uphold academic integrity, the Results chapter must explicitly state the following limitations:
1. **Sample Size:** The dataset comprises 40 total samples (24 train, 8 val, 8 test) across 10 flood events. A test set of $N=8$ is insufficient to establish statistically significant global generalizability.
2. **Synthetic SAR Proxy:** Full radar rasters from Sentinel-1 were not ingested; instead, $128 \times 128$ pixel patches were synthesized using Gaussian distributions separated by $10\text{ dB}$ ($-22\text{ dB}$ for flood, $-12\text{ dB}$ for non-flood). This substantial contrast enabled the CNN encoder to achieve trivial $100\%$ linear separation.
3. **Hindcast Reanalysis vs. Forward Forecast:** In `build_dataset.py`, the 48-hour forward weather sequence was populated using Open-Meteo ERA5 reanalysis weather (actual recorded rain during the event) rather than an archived numerical weather forecast issued 48 hours prior. The task therefore constitutes an **idealized hindcast simulation under observed meteorological forcing**, rather than an operational real-time forecast.
