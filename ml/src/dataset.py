import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from ml.src.preprocessing import MultimodalPreprocessor

class MultimodalFloodDataset(Dataset):
    """
    PyTorch Dataset for Global Multimodal Flood Prediction.
    Yields:
      - sar_patch: (2, 128, 128)
      - sar_mask: (1,)
      - hist_weather: (168, 6)
      - hist_mask: (168, 6)
      - fc_weather: (48, 6)
      - fc_mask: (48, 6)
      - geo_features: (9,)
      - geo_mask: (1,)
      - label: (1,) float (0.0 or 1.0)
    """
    def __init__(self, samples: list[dict]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.samples[idx]

        # SAR Modality
        raw_sar = item.get("sar_patch")
        if raw_sar is not None:
            sar_arr = np.array(raw_sar, dtype=np.float32)
        else:
            sar_arr = None
        sar_norm, sar_avail = MultimodalPreprocessor.preprocess_sar_patch(sar_arr)

        # Weather Modality
        hist_records = item.get("historical_weather_168h", [])
        fc_records = item.get("forecast_weather_48h", [])
        hist_val, hist_mask = MultimodalPreprocessor.preprocess_weather_sequence(hist_records, target_hours=168)
        fc_val, fc_mask = MultimodalPreprocessor.preprocess_weather_sequence(fc_records, target_hours=48)

        # Geographic Modality
        lat = item.get("latitude", 0.0)
        lon = item.get("longitude", 0.0)
        terrain = item.get("terrain_features", {})
        geo_val, geo_avail = MultimodalPreprocessor.preprocess_geo_features(lat, lon, terrain)

        label = float(item.get("target_flood_within_48h", 0.0))

        return {
            "sample_id": item.get("sample_id", f"sample_{idx}"),
            "event_id": item.get("event_id", "none"),
            "sar_patch": torch.from_numpy(sar_norm),
            "sar_mask": torch.tensor([sar_avail], dtype=torch.float32),
            "hist_weather": torch.from_numpy(hist_val),
            "hist_weather_mask": torch.from_numpy(hist_mask),
            "fc_weather": torch.from_numpy(fc_val),
            "fc_weather_mask": torch.from_numpy(fc_mask),
            "geo_features": torch.from_numpy(geo_val),
            "geo_mask": torch.tensor([geo_avail], dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.float32)
        }


def create_dataloaders(
    train_samples: list[dict],
    val_samples: list[dict],
    test_samples: list[dict],
    batch_size: int = 16,
    num_workers: int = 0
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = MultimodalFloodDataset(train_samples)
    val_ds = MultimodalFloodDataset(val_samples)
    test_ds = MultimodalFloodDataset(test_samples)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
