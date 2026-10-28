# Figure Descriptions & Scientific Documentation: GlobalFlood AI

This document provides academic descriptions, data provenance, captions, and explicit experimental limitations for each figure generated for the Results and Discussion chapter.

All figures were generated programmatically by [`ml/scripts/generate_report_figures.py`](file:///Users/jathinaraveti/Desktop/ML/GlobalFloodAI/ml/scripts/generate_report_figures.py) exclusively from existing evaluation artifacts, manifests, and checkpoints without retraining or altering any project files.

---

## Figure 1: Training and Validation Convergence History

- **Files:** `reports/figures/fig1_training_validation_loss.png` (300 DPI), `reports/figures/fig1_training_validation_loss.pdf`
- **Data Source:** [`models/checkpoints/training_history.json`](file:///Users/jathinaraveti/Desktop/ML/GlobalFloodAI/models/checkpoints/training_history.json)
- **Title:** Figure 1: MultimodalFloodNet Training Loss and Validation Tracking Across Epochs
- **Caption:** 
  *(A) Binary cross-entropy (BCE) loss progression over 5 optimization epochs for MultimodalFloodNet using the AdamW optimizer (learning rate $10^{-3}$, weight decay $10^{-4}$). The validation loss drops monotonically to its minimum of $0.0198$ at epoch 5. (B) Validation tracking for ROC-AUC and F1-score across epochs. Validation ROC-AUC remains at $1.000$ throughout, while F1-score reaches $1.000$ at epoch 2 following initial threshold adjustment.*
- **Interpretation:** 
  The network achieves rapid convergence within 5 epochs on the training split ($N=24$), with both training loss ($0.0383$) and validation loss ($0.0198$) stabilizing near zero.
- **Academic Limitations:** 
  Training accuracy and validation accuracy were not recorded in the training loop of `train.py` and are therefore omitted to avoid fabrication. The rapid convergence to near-zero loss reflects the strong separability introduced by the synthetic SAR proxy backscatter distribution.

---

## Figure 2: Confusion Matrices Across Evaluated Models

- **Files:** `reports/figures/fig2_confusion_matrices.png` (300 DPI), `reports/figures/fig2_confusion_matrices.pdf`
- **Data Source:** [`models/checkpoints/full_evaluation_metrics.json`](file:///Users/jathinaraveti/Desktop/ML/GlobalFloodAI/models/checkpoints/full_evaluation_metrics.json) (keys: `models.<name>.confusion_matrix`)
- **Title:** Figure 2: Confusion Matrices for Evaluated Models on Held-Out Test Events ($N=8$)
- **Caption:** 
  *Annotated $2 \times 2$ confusion matrices evaluated on the 8 held-out test samples (4 positive flooded points, 4 negative non-flooded references) across test events GFD_MODIS_4210 (Sylhet Basin, Bangladesh) and GFD_MODIS_3350 (Yorkshire River, United Kingdom). (A) Raw uncalibrated MultimodalFloodNet ($\text{TN}=4, \text{FP}=0, \text{FN}=0, \text{TP}=4$). (B) Platt-calibrated MultimodalFloodNet ($\text{TN}=4, \text{FP}=0, \text{FN}=0, \text{TP}=4$). (C) Tabular logistic baseline ($\text{TN}=3, \text{FP}=1, \text{FN}=0, \text{TP}=4$). (D) Prior-frequency baseline predicting training class prevalence ($\text{TN}=0, \text{FP}=4, \text{FN}=0, \text{TP}=4$).*
- **Interpretation:** 
  Both raw and calibrated versions of MultimodalFloodNet achieve zero false positives and zero false negatives at the default decision threshold of $0.50$. The tabular baseline incurs 1 false positive on negative sample `GFD_MODIS_3350_neg_1`, while the trivial prior-frequency baseline classifies all samples as positive.
- **Academic Limitations:** 
  The test set sample size is $N=8$. With only 4 samples per class, zero errors must not be interpreted as empirical proof of infallible real-world performance; it reflects the small test sample size and synthetic feature contrast.

---

## Figure 3: Receiver Operating Characteristic (ROC) Curves

- **Files:** `reports/figures/fig3_roc_curves.png` (300 DPI), `reports/figures/fig3_roc_curves.pdf`
- **Data Source:** [`models/checkpoints/full_evaluation_metrics.json`](file:///Users/jathinaraveti/Desktop/ML/GlobalFloodAI/models/checkpoints/full_evaluation_metrics.json) (`individual_sample_predictions`)
- **Title:** Figure 3: Receiver Operating Characteristic (ROC) Discrimination Curves
- **Caption:** 
  *ROC curves evaluated on the held-out test set ($N=8$). Both raw and Platt-calibrated MultimodalFloodNet achieve an empirical ROC-AUC of $1.0000$, traversing coordinates $(0,0) \to (0,1) \to (1,1)$. The diagonal dashed line denotes a random classifier ($\text{AUC} = 0.5000$). The tabular logistic baseline achieved a recorded scalar ROC-AUC of $0.9375$.*
- **Interpretation:** 
  The multimodal deep network completely discriminates positive from negative flood cases across all classification thresholds on this test partition.
- **Academic Limitations:** 
  Because continuous individual sample probabilities for the tabular baseline were not retained in the `individual_sample_predictions` JSON array, its curve cannot be plotted without fabrication; its verified scalar AUC ($0.9375$) is explicitly reported in the legend. The perfect AUC ($1.0000$) is driven by the $10\text{ dB}$ difference in synthetic SAR backscatter.

---

## Figure 4: Precision–Recall Curves

- **Files:** `reports/figures/fig4_precision_recall_curves.png` (300 DPI), `reports/figures/fig4_precision_recall_curves.pdf`
- **Data Source:** [`models/checkpoints/full_evaluation_metrics.json`](file:///Users/jathinaraveti/Desktop/ML/GlobalFloodAI/models/checkpoints/full_evaluation_metrics.json) (`individual_sample_predictions`)
- **Title:** Figure 4: Precision–Recall Curves for Flood Onset Detection
- **Caption:** 
  *Precision–Recall curves on the held-out test set ($N=8$, balanced positive prevalence $P = 0.50$). Raw and calibrated MultimodalFloodNet maintain $100\%$ precision across all recall levels ($\text{PR-AUC} = 1.0000$). The horizontal red dotted line marks the positive class baseline prevalence ($0.50$). The tabular logistic baseline recorded a PR-AUC of $0.9437$.*
- **Interpretation:** 
  The model identifies all positive flood events without returning any false alarms, maintaining maximum precision at full recall on the test partition.
- **Academic Limitations:** 
  In operational global flood forecasting, positive flood events represent extreme low-prevalence anomalies ($<2\%$), where PR-AUC typically drops due to false alarms in complex terrain. The $50\%$ balanced test prevalence here was artificially structured by paired sampling.

---

## Figure 5: Reliability Diagram (Probability Calibration)

- **Files:** `reports/figures/fig5_reliability_diagram.png` (300 DPI), `reports/figures/fig5_reliability_diagram.pdf`
- **Data Source:** [`models/checkpoints/full_evaluation_metrics.json`](file:///Users/jathinaraveti/Desktop/ML/GlobalFloodAI/models/checkpoints/full_evaluation_metrics.json) (`calibration_bins` for each model)
- **Title:** Figure 5: Reliability Diagram Comparing Raw vs. Calibrated Forecast Probabilities
- **Caption:** 
  *Empirical flood frequency plotted against mean predicted confidence across 5 equal-width bins $[0.0, 0.2), \dots, [0.8, 1.0]$. The diagonal line represents perfect calibration ($y = x$). Empty bins (count $= 0$) are omitted. (Raw Multimodal: $\text{ECE} = 0.0199, \text{Brier} = 0.0004$; Platt Calibrated: $\text{ECE} = 0.3458, \text{Brier} = 0.1196$; Tabular Baseline: $\text{ECE} = 0.0967, \text{Brier} = 0.0908$).*
- **Critical Scientific Finding:** 
  Platt scaling **worsened** calibration metrics on the test set. In the raw model, predictions were tightly concentrated near ground truth (mean confidence $0.0203$ for non-floods, $0.9805$ for floods), yielding low ECE ($0.0199$) and near-zero Brier score ($0.0004$). Platt scaling shifted probabilities inward toward the prior mean ($0.3461$ and $0.6546$), introducing artificial calibration gaps ($0.346$) and inflating Brier score by two orders of magnitude to $0.1196$.
- **Academic Limitations:** 
  A test set of 8 samples cannot reliably parameterize or validate a 5-bin calibration curve. This phenomenon demonstrates the risk of fitting parametric post-hoc calibration on small sample datasets.

---

## Figure 6: Dataset Partitioning and Geographic Allocation

- **Files:** `reports/figures/fig6_dataset_split_distribution.png` (300 DPI), `reports/figures/fig6_dataset_split_distribution.pdf`
- **Data Source:** [`data/processed/dataset_manifest.json`](file:///Users/jathinaraveti/Desktop/ML/GlobalFloodAI/data/processed/dataset_manifest.json) & [`dataset_summary.json`](file:///Users/jathinaraveti/Desktop/ML/GlobalFloodAI/data/processed/dataset_summary.json)
- **Title:** Figure 6: Dataset Split Distribution and Event-Grouped Isolation Protocol
- **Caption:** 
  *(A) Distribution of multimodal samples across Train ($N=24$), Validation ($N=8$), and Test ($N=8$) partitions, with exactly $50\%$ flood and $50\%$ non-flood points per split. (B) Geographic allocation of the 10 cataloged global flood events across partitions, demonstrating strict event-grouped isolation where all points for a given event remain in a single split to eliminate spatial-temporal leakage.*
- **Interpretation:** 
  The dataset balances classes evenly across all partitions while maintaining strict geographical disjointness across 6 continents (Asia, Europe, North America, South America, Africa, Australia).
- **Academic Limitations:** 
  The total volume ($N=40$ samples) is an experimental prototype dataset. Operational deep learning benchmarks (e.g., Sen1Floods11) utilize tens of thousands of image chips.

---

## Figure 7: Comprehensive Model Benchmark Comparison

- **Files:** `reports/figures/fig7_model_comparison.png` (300 DPI), `reports/figures/fig7_model_comparison.pdf`
- **Data Source:** [`models/checkpoints/full_evaluation_metrics.json`](file:///Users/jathinaraveti/Desktop/ML/GlobalFloodAI/models/checkpoints/full_evaluation_metrics.json)
- **Title:** Figure 7: Comprehensive Benchmark of Evaluated Flood Prediction Models
- **Caption:** 
  *(A) Discrimination and classification metrics (ROC-AUC, PR-AUC, F1-Score, and Accuracy at default threshold $0.50$) across four evaluated models: MultimodalFloodNet Raw, MultimodalFloodNet Platt Calibrated, Tabular Logistic Baseline, and Prior Frequency Baseline. Higher values indicate superior performance. (B) Probabilistic reliability measured by Brier score (lower values indicate superior probability accuracy).*
- **Interpretation:** 
  Multimodal deep learning outperforms both the tabular baseline (accuracy $0.875$, F1 $0.8889$) and the trivial prior baseline (accuracy $0.500$, F1 $0.6667$). The raw multimodal network achieves the lowest Brier score ($0.0004$).
- **Academic Limitations:** 
  The tabular baseline utilizes aggregated rainfall and terrain statistics but lacks high-resolution spatial and sequential temporal dynamics. Its strong performance (ROC-AUC $0.9375$) indicates that rainfall volume and slope alone provide substantial predictive signal in this synthesized dataset.
