# GlobalFlood AI Models & Checkpoints

This directory houses PyTorch state dictionaries, calibrated scalers, and baseline model artifacts.

## Checkpoints
- `checkpoints/multimodal_flood_net_best.pt`: Best model weights for `MultimodalFloodNet` containing:
  - `model_state_dict`: Weights for SatelliteImageEncoder (128-d), WeatherTimeSeriesEncoder (64-d), GeographicFeatureEncoder (32-d), and Fusion layers.
  - `val_loss`: Held-out validation BCE loss.
  - `val_auc`: Validation ROC-AUC.
  - `is_smoke_test`: Boolean indicating whether trained on synthetic smoke data or aligned historical archives.
- `checkpoints/calibrator.joblib`: Serialized `ProbabilityCalibrator` using Platt scaling (sigmoid mapping) on validation logits.
- `checkpoints/baseline_xgboost.joblib`: Logistic Regression / Tabular Baseline classifier for benchmark comparisons.
- `checkpoints/training_history.json`: Epoch-by-epoch training and validation loss curves.
