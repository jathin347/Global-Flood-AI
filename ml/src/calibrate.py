import numpy as np
import joblib
from pathlib import Path
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from app.core.logging import logger

class ProbabilityCalibrator:
    """
    Fits probability calibration on held-out validation predictions
    using Platt Scaling (sigmoid) or Isotonic Regression.
    Maps raw uncalibrated network scores to calibrated empirical frequencies.
    """
    def __init__(self, method: str = "platt"):
        self.method = method
        if method == "isotonic":
            self.calibrator = IsotonicRegression(out_of_bounds="clip")
        else:
            self.calibrator = LogisticRegression()
        self.is_fitted = False

    def fit(self, val_raw_probs: np.ndarray, val_labels: np.ndarray):
        if len(np.unique(val_labels)) < 2:
            logger.warning("Calibration requires at least two distinct classes in validation set.")
            return

        if self.method == "isotonic":
            self.calibrator.fit(val_raw_probs, val_labels)
        else:
            X = val_raw_probs.reshape(-1, 1)
            self.calibrator.fit(X, val_labels)
        self.is_fitted = True
        logger.info(f"Calibrator successfully fitted using {self.method} method.")

    def calibrate(self, raw_prob: float) -> float:
        if not self.is_fitted:
            return float(raw_prob)
        try:
            if self.method == "isotonic":
                cal = float(self.calibrator.predict([raw_prob])[0])
            else:
                cal = float(self.calibrator.predict_proba([[raw_prob]])[0, 1])
            return float(np.clip(cal, 0.0, 1.0))
        except Exception:
            return float(raw_prob)

    def save(self, filepath: Path):
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "method": self.method,
            "calibrator": self.calibrator,
            "is_fitted": self.is_fitted
        }, filepath)

    @classmethod
    def load(cls, filepath: Path) -> "ProbabilityCalibrator":
        if filepath.exists():
            try:
                data = joblib.load(filepath)
                if isinstance(data, dict):
                    obj = cls(method=data.get("method", "platt"))
                    obj.calibrator = data["calibrator"]
                    obj.is_fitted = data.get("is_fitted", True)
                    return obj
                elif hasattr(data, "calibrate"):
                    return data
            except Exception as e:
                logger.warning(f"Error loading calibrator from {filepath}: {e}")
        return cls()



def main():
    import json
    import torch
    from torch.utils.data import DataLoader
    from app.core.config import settings
    from ml.src.model import MultimodalFloodNet
    from ml.src.dataset import MultimodalFloodDataset

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    weights_path = settings.MODEL_WEIGHTS_FILE

    if not weights_path.exists():
        logger.error(f"Checkpoint not found at {weights_path}. Train model first.")
        return

    ckpt = torch.load(weights_path, map_location=device)
    model = MultimodalFloodNet().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    val_file = settings.PROCESSED_DATA_DIR / "val_samples.json"
    if val_file.exists():
        with open(val_file) as f:
            val_samples = json.load(f)
    else:
        from ml.src.data_collection import create_smoke_test_dataset
        _, val_samples, _ = create_smoke_test_dataset(num_samples=32)

    val_ds = MultimodalFloodDataset(val_samples)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)
    val_probs, val_labels = [], []

    with torch.no_grad():
        for b in val_loader:
            p = torch.sigmoid(model(
                b["sar_patch"].to(device), b["sar_mask"].to(device),
                b["hist_weather"].to(device), b["hist_weather_mask"].to(device),
                b["fc_weather"].to(device), b["fc_weather_mask"].to(device),
                b["geo_features"].to(device), b["geo_mask"].to(device)
            )).cpu().numpy()
            val_probs.extend(p.tolist())
            val_labels.extend(b["label"].numpy().tolist())

    calibrator = ProbabilityCalibrator(method="platt")
    calibrator.fit(np.array(val_probs), np.array(val_labels))
    calibrator.save(settings.CALIBRATION_FILE)
    logger.info(f"Fitted and saved probability calibrator to {settings.CALIBRATION_FILE}")


if __name__ == "__main__":
    main()

