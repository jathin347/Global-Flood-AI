import torch
import torch.nn as nn

class SatelliteImageEncoder(nn.Module):
    """
    Branch 1: Satellite SAR Image Patch Encoder
    Input: (B, 2, 128, 128) - Sentinel-1 SAR VV and VH polarizations (dB normalized)
    Output: 128-dimensional embedding
    """
    def __init__(self, in_channels: int = 2, embedding_dim: int = 128):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 64x64
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 32x32
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 16x16
        )
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))  # Global Average Pooling -> (B, 128, 1, 1)
        )
        self.fc_proj = nn.Sequential(
            nn.Linear(128, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 2, 128, 128)
        h = self.conv_block1(x)
        h = self.conv_block2(h)
        h = self.conv_block3(h)
        h = self.conv_block4(h)
        h = torch.flatten(h, 1)
        return self.fc_proj(h)


class WeatherTimeSeriesEncoder(nn.Module):
    """
    Branch 2: Weather Time-Series Encoder
    Inputs:
      - Historical sequence: (B, 168, num_features) with precipitation, temp, humidity, pressure, wind, soil
      - Forecast sequence: (B, 48, num_features)
    Output: 64-dimensional embedding
    Uses GRU cells + missingness mask awareness.
    """
    def __init__(self, feature_dim: int = 6, hidden_dim: int = 64, embedding_dim: int = 64):
        super().__init__()
        # Input includes values + binary missingness indicators -> feature_dim * 2
        in_dim = feature_dim * 2
        self.hist_gru = nn.GRU(
            input_size=in_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=False,
            dropout=0.1
        )
        self.fc_gru = nn.GRU(
            input_size=in_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )
        self.fusion_fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, hist_seq: torch.Tensor, hist_mask: torch.Tensor,
                fc_seq: torch.Tensor, fc_mask: torch.Tensor) -> torch.Tensor:
        # hist_seq: (B, 168, F), hist_mask: (B, 168, F)
        hist_input = torch.cat([hist_seq, hist_mask], dim=-1)
        _, h_hist = self.hist_gru(hist_input)  # h_hist: (2, B, hidden_dim)
        hist_rep = h_hist[-1]  # (B, hidden_dim)

        fc_input = torch.cat([fc_seq, fc_mask], dim=-1)
        _, h_fc = self.fc_gru(fc_input)
        fc_rep = h_fc[-1]  # (B, hidden_dim)

        combined = torch.cat([hist_rep, fc_rep], dim=-1)
        return self.fusion_fc(combined)


class GeographicFeatureEncoder(nn.Module):
    """
    Branch 3: Topographic and Geographic Feature Encoder
    Input: (B, num_features) - elevation, slope, TRI, TWI, sin(lat), cos(lat), sin(lon), cos(lon), dist_river
    Output: 32-dimensional embedding
    """
    def __init__(self, in_features: int = 9, embedding_dim: int = 32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class MultimodalFloodNet(nn.Module):
    """
    Multimodal Deep Learning System for 48-Hour Flood Forecasting
    Combines:
      - Sentinel-1 SAR Image Patch (128-d)
      - Weather 168h Historical + 48h Forecast GRU (64-d)
      - Topographic / Geographic MLP (32-d)
      Total combined representation: 224-d
    Supports missing modality indicators & mask-aware fusion.
    """
    def __init__(
        self,
        sar_channels: int = 2,
        weather_features: int = 6,
        geo_features: int = 9,
        sar_embed_dim: int = 128,
        weather_embed_dim: int = 64,
        geo_embed_dim: int = 32
    ):
        super().__init__()
        self.sar_encoder = SatelliteImageEncoder(in_channels=sar_channels, embedding_dim=sar_embed_dim)
        self.weather_encoder = WeatherTimeSeriesEncoder(feature_dim=weather_features, embedding_dim=weather_embed_dim)
        self.geo_encoder = GeographicFeatureEncoder(in_features=geo_features, embedding_dim=geo_embed_dim)

        # Modality presence indicators (3 binary flags: [has_sar, has_weather, has_geo])
        fusion_dim = sar_embed_dim + weather_embed_dim + geo_embed_dim + 3

        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 1)  # Logit for flood_within_48h
        )

    def forward(
        self,
        sar_patches: torch.Tensor,
        sar_mask: torch.Tensor,
        hist_weather: torch.Tensor,
        hist_weather_mask: torch.Tensor,
        fc_weather: torch.Tensor,
        fc_weather_mask: torch.Tensor,
        geo_features: torch.Tensor,
        geo_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        sar_mask: (B, 1) - 1.0 if SAR available, 0.0 otherwise
        geo_mask: (B, 1) - 1.0 if Geo available, 0.0 otherwise
        """
        B = hist_weather.size(0)

        # Encode SAR branch (zeroed if unavailable)
        sar_raw = self.sar_encoder(sar_patches)
        sar_rep = sar_raw * sar_mask.view(B, 1)

        # Encode Weather branch
        # Weather availability flag = 1.0 if weather sequence has valid elements
        weather_mask_flag = (hist_weather_mask.sum(dim=(1, 2)) > 0).float().view(B, 1)
        weather_rep = self.weather_encoder(hist_weather, hist_weather_mask, fc_weather, fc_weather_mask)

        # Encode Geographic branch
        geo_raw = self.geo_encoder(geo_features)
        geo_rep = geo_raw * geo_mask.view(B, 1)

        # Modality availability vector: (B, 3)
        modality_flags = torch.cat([sar_mask.view(B, 1), weather_mask_flag, geo_mask.view(B, 1)], dim=-1)

        # Concatenate 128 + 64 + 32 + 3 = 227 dimensions
        fused = torch.cat([sar_rep, weather_rep, geo_rep, modality_flags], dim=-1)
        logits = self.fusion(fused)
        return logits.squeeze(-1)

    def predict_probability(self, *args, **kwargs) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            logits = self.forward(*args, **kwargs)
            return torch.sigmoid(logits)


class WeatherOnlyGRU(nn.Module):
    """
    Ablation / Baseline: Weather time-series only model
    """
    def __init__(self, feature_dim: int = 6, hidden_dim: int = 64):
        super().__init__()
        self.encoder = WeatherTimeSeriesEncoder(feature_dim=feature_dim, embedding_dim=hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1)
        )

    def forward(self, hist_seq, hist_mask, fc_seq, fc_mask):
        rep = self.encoder(hist_seq, hist_mask, fc_seq, fc_mask)
        return self.classifier(rep).squeeze(-1)
