import json
import random
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score
import joblib

from ml.src.model import MultimodalFloodNet
from ml.src.dataset import create_dataloaders
from app.core.config import settings
from app.core.logging import logger

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def train_multimodal_model(
    train_samples: list[dict],
    val_samples: list[dict],
    epochs: int = 15,
    lr: float = 1e-3,
    batch_size: int = 8,
    save_path: Optional[Path] = None,
    is_smoke_test: bool = True
) -> dict:
    set_seed(42)
    device = get_device()
    logger.info(f"Starting training on device: {device} | Smoke test: {is_smoke_test}")

    train_loader, val_loader, _ = create_dataloaders(train_samples, val_samples, [], batch_size=batch_size)

    model = MultimodalFloodNet().to(device)

    # Class imbalance handling
    labels = [s["target_flood_within_48h"] for s in train_samples]
    pos_count = sum(labels)
    neg_count = len(labels) - pos_count
    pos_weight_val = max(0.1, neg_count / max(1.0, pos_count))
    pos_weight = torch.tensor([pos_weight_val], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": [], "val_auc": [], "val_f1": []}

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            sar = batch["sar_patch"].to(device)
            sar_m = batch["sar_mask"].to(device)
            hw = batch["hist_weather"].to(device)
            hw_m = batch["hist_weather_mask"].to(device)
            fw = batch["fc_weather"].to(device)
            fw_m = batch["fc_weather_mask"].to(device)
            geo = batch["geo_features"].to(device)
            geo_m = batch["geo_mask"].to(device)
            target = batch["label"].to(device)

            logits = model(sar, sar_m, hw, hw_m, fw, fw_m, geo, geo_m)
            loss = criterion(logits, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * sar.size(0)

        train_loss /= len(train_samples)

        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for batch in val_loader:
                sar = batch["sar_patch"].to(device)
                sar_m = batch["sar_mask"].to(device)
                hw = batch["hist_weather"].to(device)
                hw_m = batch["hist_weather_mask"].to(device)
                fw = batch["fc_weather"].to(device)
                fw_m = batch["fc_weather_mask"].to(device)
                geo = batch["geo_features"].to(device)
                geo_m = batch["geo_mask"].to(device)
                target = batch["label"].to(device)

                logits = model(sar, sar_m, hw, hw_m, fw, fw_m, geo, geo_m)
                v_loss = criterion(logits, target)
                val_loss += v_loss.item() * sar.size(0)

                probs = torch.sigmoid(logits).cpu().numpy()
                all_preds.extend(probs.tolist())
                all_targets.extend(target.cpu().numpy().tolist())

        val_loss /= len(val_samples)
        scheduler.step(val_loss)

        # Metrics
        y_true = np.array(all_targets)
        y_pred_prob = np.array(all_preds)
        y_pred_bin = (y_pred_prob >= 0.5).astype(int)

        try:
            val_auc = float(roc_auc_score(y_true, y_pred_prob)) if len(np.unique(y_true)) > 1 else 0.5
        except Exception:
            val_auc = 0.5
        val_f1 = float(f1_score(y_true, y_pred_bin, zero_division=0))

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)
        history["val_f1"].append(val_f1)

        logger.info(f"Epoch {epoch:02d}/{epochs:02d} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val AUC: {val_auc:.4f} - Val F1: {val_f1:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if save_path:
                save_path.parent.mkdir(parents=True, exist_ok=True)
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_auc": val_auc,
                    "is_smoke_test": is_smoke_test,
                    "architecture": "MultimodalFloodNet (SAR 128d + Weather GRU 64d + Geo MLP 32d + Fusion)"
                }
                torch.save(checkpoint, save_path)
                logger.info(f"Saved best model checkpoint to {save_path}")

    # Save training history JSON
    history_path = (save_path.parent if save_path else settings.CHECKPOINTS_DIR) / "training_history.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    return {"history": history, "best_val_loss": best_val_loss, "model": model}


def train_baseline_model(train_samples: list[dict], val_samples: list[dict], save_path: Path):
    """
    Trains a Logistic Regression / Tabular Baseline model using weather summary + terrain features.
    """
    def extract_tabular_features(samples: list[dict]) -> tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for s in samples:
            hw = s.get("historical_weather_168h", [])
            fw = s.get("forecast_weather_48h", [])
            terr = s.get("terrain_features", {})

            # Aggregate features: 168h total rain, 48h total forecast rain, max rain, soil, elev, slope
            hist_rain = sum(r.get("precipitation_mm", 0.0) or 0.0 for r in hw)
            fc_rain = sum(r.get("precipitation_mm", 0.0) or 0.0 for r in fw)
            max_rain = max([r.get("precipitation_mm", 0.0) or 0.0 for r in (hw + fw)] or [0.0])
            elev = terr.get("elevation_m", 15.0)
            slope = terr.get("slope_degrees", 2.0)
            twi = terr.get("topographic_wetness_index", 7.0)

            X.append([hist_rain, fc_rain, max_rain, elev, slope, twi])
            y.append(s.get("target_flood_within_48h", 0.0))
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    X_train, y_train = extract_tabular_features(train_samples)
    X_val, y_val = extract_tabular_features(val_samples)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    val_preds = clf.predict_proba(X_val)[:, 1] if len(np.unique(y_train)) > 1 else np.zeros(len(y_val))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, save_path)
    logger.info(f"Trained and saved baseline classifier to {save_path}")
    return clf


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train MultimodalFloodNet model")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--data-dir", type=str, default=str(settings.PROCESSED_DATA_DIR), help="Data directory")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    train_file = data_dir / "train_samples.json"
    val_file = data_dir / "val_samples.json"

    if train_file.exists() and val_file.exists():
        logger.info(f"Loading dataset from {data_dir}...")
        with open(train_file) as f:
            train_samples = json.load(f)
        with open(val_file) as f:
            val_samples = json.load(f)
        is_smoke = False
    else:
        logger.info("Processed dataset not found, generating synthetic smoke test samples...")
        from ml.src.data_collection import create_smoke_test_dataset
        train_samples, val_samples, _ = create_smoke_test_dataset(num_samples=32)
        is_smoke = True

    train_multimodal_model(
        train_samples=train_samples,
        val_samples=val_samples,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        save_path=settings.MODEL_WEIGHTS_FILE,
        is_smoke_test=is_smoke
    )


if __name__ == "__main__":
    main()

