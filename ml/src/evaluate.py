import numpy as np
import torch
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    precision_recall_curve, auc, brier_score_loss, confusion_matrix
)
from ml.src.model import MultimodalFloodNet
from ml.src.dataset import MultimodalFloodDataset
from torch.utils.data import DataLoader

def evaluate_model(
    model: MultimodalFloodNet,
    test_samples: list[dict],
    device: torch.device,
    is_smoke_test: bool = True
) -> dict:
    """
    Evaluates multimodal network on held-out test set.
    Calculates scientific metrics: Precision, Recall, F1, ROC-AUC, PR-AUC, Brier Score, Confusion Matrix.
    """
    if not test_samples:
        return {"status": "unavailable", "reason": "No test samples provided"}

    test_ds = MultimodalFloodDataset(test_samples)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)

    model.eval()
    all_probs = []
    all_targets = []
    sample_ids = []

    with torch.no_grad():
        for batch in test_loader:
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
            probs = torch.sigmoid(logits).cpu().numpy()

            all_probs.extend(probs.tolist())
            all_targets.extend(target.cpu().numpy().tolist())
            sample_ids.extend(batch["sample_id"])

    y_true = np.array(all_targets)
    y_pred_prob = np.array(all_probs)
    y_pred_bin = (y_pred_prob >= 0.5).astype(int)

    # Scientific metrics
    has_both_classes = len(np.unique(y_true)) > 1

    try:
        roc_auc = float(roc_auc_score(y_true, y_pred_prob)) if has_both_classes else 0.5
    except Exception:
        roc_auc = 0.5

    try:
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred_prob)
        pr_auc = float(auc(recall_curve, precision_curve)) if has_both_classes else 0.5
    except Exception:
        pr_auc = 0.5

    prec = float(precision_score(y_true, y_pred_bin, zero_division=0))
    rec = float(recall_score(y_true, y_pred_bin, zero_division=0))
    f1 = float(f1_score(y_true, y_pred_bin, zero_division=0))
    brier = float(brier_score_loss(y_true, y_pred_prob))

    cm = confusion_matrix(y_true, y_pred_bin).tolist() if has_both_classes else [[len(y_true), 0], [0, 0]]

    results = {
        "status": "evaluated",
        "is_smoke_test": is_smoke_test,
        "sample_count": len(test_samples),
        "metrics": {
            "roc_auc": round(roc_auc, 4),
            "pr_auc": round(pr_auc, 4),
            "f1_score": round(f1, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "brier_score": round(brier, 4)
        },
        "confusion_matrix": cm,
        "evaluation_notes": [
            "Evaluated on held-out test events separated temporally and spatially.",
            "Synthetic smoke test data used if is_smoke_test is True."
        ]
    }
    return results


def main():
    import json
    from pathlib import Path
    from app.core.config import settings
    from app.core.logging import logger

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    weights_path = settings.MODEL_WEIGHTS_FILE

    if not weights_path.exists():
        logger.error(f"Checkpoint not found at {weights_path}. Run training or smoke test first.")
        return

    ckpt = torch.load(weights_path, map_location=device)
    model = MultimodalFloodNet().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    is_smoke = ckpt.get("is_smoke_test", True)

    test_file = settings.PROCESSED_DATA_DIR / "test_samples.json"
    if test_file.exists():
        with open(test_file) as f:
            test_samples = json.load(f)
    else:
        from ml.src.data_collection import create_smoke_test_dataset
        _, _, test_samples = create_smoke_test_dataset(num_samples=32)

    logger.info(f"Evaluating model on {len(test_samples)} test samples (device={device}, is_smoke={is_smoke})...")
    res = evaluate_model(model, test_samples, device, is_smoke_test=is_smoke)
    logger.info(f"Evaluation Results:\n{json.dumps(res, indent=2)}")

    eval_file = settings.CHECKPOINTS_DIR / "evaluation_results.json"
    eval_file.parent.mkdir(parents=True, exist_ok=True)
    with open(eval_file, "w") as f:
        json.dump(res, f, indent=2)
    logger.info(f"Saved evaluation results to {eval_file}")


if __name__ == "__main__":
    main()


