import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_dir / "backend"))
sys.path.insert(0, str(root_dir))

from ml.src.data_collection import create_smoke_test_dataset
from ml.src.train import train_multimodal_model, train_baseline_model, set_seed, get_device
from ml.src.evaluate import evaluate_model
from ml.src.calibrate import ProbabilityCalibrator
from app.core.config import settings
from app.core.logging import logger

def main():
    logger.info("==================================================")
    logger.info("GlobalFlood AI — Synthetic Architecture Smoke Test")
    logger.info("==================================================")
    set_seed(42)

    train_samples, val_samples, test_samples = create_smoke_test_dataset(num_samples=32)
    logger.info(f"Generated synthetic smoke samples: Train={len(train_samples)}, Val={len(val_samples)}, Test={len(test_samples)}")

    checkpoint_path = settings.MODEL_WEIGHTS_FILE
    res = train_multimodal_model(
        train_samples=train_samples,
        val_samples=val_samples,
        epochs=3,
        lr=1e-3,
        batch_size=4,
        save_path=checkpoint_path,
        is_smoke_test=True
    )

    model = res["model"]
    device = get_device()

    # Fit probability calibrator on validation samples
    calibrator = ProbabilityCalibrator(method="platt")
    # Generate val predictions
    from ml.src.dataset import MultimodalFloodDataset
    from torch.utils.data import DataLoader
    import torch
    import numpy as np

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

    calibrator.fit(np.array(val_probs), np.array(val_labels))
    calibrator.save(settings.CALIBRATION_FILE)
    logger.info(f"Saved calibrator to {settings.CALIBRATION_FILE}")

    # Train baseline model
    train_baseline_model(train_samples, val_samples, settings.BASELINE_MODEL_FILE)

    # Evaluate on held-out test events
    eval_res = evaluate_model(model, test_samples, device, is_smoke_test=True)
    logger.info(f"Evaluation metrics on test set: {eval_res['metrics']}")
    logger.info("Smoke test completed successfully!")

if __name__ == "__main__":
    main()
