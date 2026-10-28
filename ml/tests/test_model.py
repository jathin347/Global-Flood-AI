import tempfile
from pathlib import Path
import numpy as np
import pytest
import torch
import torch.nn as nn

from ml.src.model import (
    SatelliteImageEncoder, WeatherTimeSeriesEncoder, GeographicFeatureEncoder,
    MultimodalFloodNet, WeatherOnlyGRU
)
from ml.src.preprocessing import MultimodalPreprocessor
from ml.src.dataset import MultimodalFloodDataset
from ml.src.calibrate import ProbabilityCalibrator
from ml.src.build_dataset import validate_coordinates


def test_coordinate_validation():
    # Valid coordinates
    assert validate_coordinates(0.0, 0.0) is True
    assert validate_coordinates(13.0827, 80.2707) is True
    assert validate_coordinates(-33.8688, 151.2093) is True
    assert validate_coordinates(90.0, 180.0) is True
    assert validate_coordinates(-90.0, -180.0) is True

    # Invalid coordinates
    assert validate_coordinates(91.0, 0.0) is False
    assert validate_coordinates(-90.1, 0.0) is False
    assert validate_coordinates(0.0, 180.5) is False
    assert validate_coordinates(0.0, -181.0) is False
    assert validate_coordinates(float("nan"), 0.0) is False
    assert validate_coordinates(0.0, float("inf")) is False


def test_satellite_encoder_shapes():
    encoder = SatelliteImageEncoder(in_channels=2, embedding_dim=128)
    x = torch.randn(4, 2, 128, 128)
    out = encoder(x)
    assert out.shape == (4, 128)


def test_weather_encoder_shapes():
    encoder = WeatherTimeSeriesEncoder(feature_dim=6, embedding_dim=64)
    hist = torch.randn(4, 168, 6)
    hist_m = torch.ones(4, 168, 6)
    fc = torch.randn(4, 48, 6)
    fc_m = torch.ones(4, 48, 6)
    out = encoder(hist, hist_m, fc, fc_m)
    assert out.shape == (4, 64)


def test_geo_encoder_shapes():
    encoder = GeographicFeatureEncoder(in_features=9, embedding_dim=32)
    x = torch.randn(4, 9)
    out = encoder(x)
    assert out.shape == (4, 32)


def test_multimodal_flood_net_forward():
    model = MultimodalFloodNet()
    B = 2
    sar = torch.randn(B, 2, 128, 128)
    sar_m = torch.ones(B, 1)
    hw = torch.randn(B, 168, 6)
    hw_m = torch.ones(B, 168, 6)
    fw = torch.randn(B, 48, 6)
    fw_m = torch.ones(B, 48, 6)
    geo = torch.randn(B, 9)
    geo_m = torch.ones(B, 1)

    logits = model(sar, sar_m, hw, hw_m, fw, fw_m, geo, geo_m)
    assert logits.shape == (B,)

    probs = model.predict_probability(sar, sar_m, hw, hw_m, fw, fw_m, geo, geo_m)
    assert probs.shape == (B,)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()


def test_missing_modality_handling():
    """Verify model gracefully processes missing SAR, missing weather, and missing terrain."""
    model = MultimodalFloodNet()
    model.eval()
    B = 2

    # 1. Missing SAR (mask=0)
    sar_zero = torch.zeros(B, 2, 128, 128)
    sar_m_zero = torch.zeros(B, 1)
    hw = torch.randn(B, 168, 6)
    hw_m = torch.ones(B, 168, 6)
    fw = torch.randn(B, 48, 6)
    fw_m = torch.ones(B, 48, 6)
    geo = torch.randn(B, 9)
    geo_m = torch.ones(B, 1)

    logits_no_sar = model(sar_zero, sar_m_zero, hw, hw_m, fw, fw_m, geo, geo_m)
    assert logits_no_sar.shape == (B,)
    assert not torch.isnan(logits_no_sar).any()

    # 2. Missing Weather (mask=0)
    hw_zero = torch.zeros(B, 168, 6)
    hw_m_zero = torch.zeros(B, 168, 6)
    fw_zero = torch.zeros(B, 48, 6)
    fw_m_zero = torch.zeros(B, 48, 6)

    logits_no_weather = model(sar_zero, sar_m_zero, hw_zero, hw_m_zero, fw_zero, fw_m_zero, geo, geo_m)
    assert logits_no_weather.shape == (B,)
    assert not torch.isnan(logits_no_weather).any()

    # 3. Missing Terrain (mask=0)
    geo_zero = torch.zeros(B, 9)
    geo_m_zero = torch.zeros(B, 1)
    logits_all_missing = model(sar_zero, sar_m_zero, hw_zero, hw_m_zero, fw_zero, fw_m_zero, geo_zero, geo_m_zero)
    assert logits_all_missing.shape == (B,)
    assert not torch.isnan(logits_all_missing).any()


def test_preprocessing_missing_sar():
    patch, avail = MultimodalPreprocessor.preprocess_sar_patch(None)
    assert patch.shape == (2, 128, 128)
    assert avail == 0.0
    assert np.all(patch == 0.0)


def test_preprocessing_weather_normalization():
    records = [{"precipitation_mm": 10.0, "temperature_2m_c": 25.0}]
    vals, mask = MultimodalPreprocessor.preprocess_weather_sequence(records, target_hours=10)
    assert vals.shape == (10, 6)
    assert mask.shape == (10, 6)
    # The last element should have mask = 1.0 for valid fields
    assert mask[-1, 0] == 1.0
    assert mask[-1, 1] == 1.0
    # Missing fields should have mask = 0.0
    assert mask[-1, 2] == 0.0


def test_preprocessing_geo_features():
    terrain = {"elevation_m": 45.0, "slope_degrees": 3.2, "terrain_roughness_index": 5.0}
    vec, mask = MultimodalPreprocessor.preprocess_geo_features(13.0827, 80.2707, terrain)
    assert vec.shape == (9,)
    assert mask == 1.0
    # Values should be finite
    assert np.all(np.isfinite(vec))


def test_dataset_loading_and_labels():
    sample = {
        "sample_id": "test_001",
        "event_id": "TEST_EVENT",
        "latitude": 13.0,
        "longitude": 80.0,
        "target_flood_within_48h": 1.0,
        "sar_patch": np.zeros((2, 128, 128)).tolist(),
        "historical_weather_168h": [{"precipitation_mm": 1.0}],
        "forecast_weather_48h": [{"precipitation_mm": 2.0}],
        "terrain_features": {"elevation_m": 10.0, "slope_degrees": 1.5}
    }
    ds = MultimodalFloodDataset([sample])
    assert len(ds) == 1
    item = ds[0]
    assert item["sar_patch"].shape == (2, 128, 128)
    assert item["hist_weather"].shape == (168, 6)
    assert item["fc_weather"].shape == (48, 6)
    assert item["geo_features"].shape == (9,)
    assert item["label"].item() == 1.0


def test_training_loss_and_backprop():
    model = MultimodalFloodNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    sar = torch.randn(2, 2, 128, 128)
    sar_m = torch.ones(2, 1)
    hw = torch.randn(2, 168, 6)
    hw_m = torch.ones(2, 168, 6)
    fw = torch.randn(2, 48, 6)
    fw_m = torch.ones(2, 48, 6)
    geo = torch.randn(2, 9)
    geo_m = torch.ones(2, 1)
    target = torch.tensor([1.0, 0.0], dtype=torch.float32)

    optimizer.zero_grad()
    logits = model(sar, sar_m, hw, hw_m, fw, fw_m, geo, geo_m)
    loss = criterion(logits, target)
    loss.backward()

    # Check gradients exist and are finite
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"Gradient missing for {name}"
            assert not torch.isnan(p.grad).any(), f"NaN gradient in {name}"

    optimizer.step()
    assert loss.item() > 0.0


def test_checkpoint_saving_and_loading():
    model1 = MultimodalFloodNet()
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "test_ckpt.pt"
        torch.save({
            "epoch": 5,
            "model_state_dict": model1.state_dict(),
            "val_loss": 0.42,
            "is_smoke_test": True
        }, ckpt_path)

        assert ckpt_path.exists()

        model2 = MultimodalFloodNet()
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model2.load_state_dict(ckpt["model_state_dict"])
        assert ckpt["epoch"] == 5
        assert ckpt["val_loss"] == 0.42

        # Check identical output on same input
        model1.eval()
        model2.eval()
        with torch.no_grad():
            sar = torch.randn(1, 2, 128, 128)
            sar_m = torch.ones(1, 1)
            hw = torch.randn(1, 168, 6)
            hw_m = torch.ones(1, 168, 6)
            fw = torch.randn(1, 48, 6)
            fw_m = torch.ones(1, 48, 6)
            geo = torch.randn(1, 9)
            geo_m = torch.ones(1, 1)

            out1 = model1(sar, sar_m, hw, hw_m, fw, fw_m, geo, geo_m)
            out2 = model2(sar, sar_m, hw, hw_m, fw, fw_m, geo, geo_m)
            assert torch.allclose(out1, out2)


def test_probability_calibrator():
    calibrator = ProbabilityCalibrator(method="platt")
    raw_probs = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    labels = np.array([0, 0, 0, 1, 1, 1])

    calibrator.fit(raw_probs, labels)
    assert calibrator.is_fitted is True

    # Monotonicity check
    p_low = calibrator.calibrate(0.15)
    p_high = calibrator.calibrate(0.85)
    assert 0.0 <= p_low <= 1.0
    assert 0.0 <= p_high <= 1.0
    assert p_high > p_low

    # Save and reload
    with tempfile.TemporaryDirectory() as tmpdir:
        calib_path = Path(tmpdir) / "calibrator.joblib"
        calibrator.save(calib_path)
        loaded = ProbabilityCalibrator.load(calib_path)
        assert loaded.is_fitted is True
        assert np.isclose(loaded.calibrate(0.5), calibrator.calibrate(0.5))


def test_weather_only_baseline_model():
    model = WeatherOnlyGRU(feature_dim=6, hidden_dim=32)
    hist = torch.randn(2, 168, 6)
    hist_m = torch.ones(2, 168, 6)
    fc = torch.randn(2, 48, 6)
    fc_m = torch.ones(2, 48, 6)
    logits = model(hist, hist_m, fc, fc_m)
    assert logits.shape == (2,)
