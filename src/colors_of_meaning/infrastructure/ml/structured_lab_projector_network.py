import torch
import torch.nn as nn


class StructuredLabProjectorNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int = 384,
        hidden_dim_1: int = 128,
        hidden_dim_2: int = 64,
        dropout_rate: float = 0.1,
        max_chroma: float = 128.0,
    ) -> None:
        super().__init__()

        self.max_chroma = max_chroma

        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        self.lightness_head = nn.Sequential(
            nn.Linear(hidden_dim_2, 1),
            nn.Sigmoid(),
        )

        self.hue_head = nn.Linear(hidden_dim_2, 2)

        self.chroma_head = nn.Sequential(
            nn.Linear(hidden_dim_2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)

        lightness = self.lightness_head(features) * 100.0

        hue_components = self.hue_head(features)
        hue_norm = torch.nn.functional.normalize(hue_components, dim=1)

        chroma = self.chroma_head(features) * self.max_chroma

        a_val = chroma * hue_norm[:, 0:1]
        b_val = chroma * hue_norm[:, 1:2]

        return torch.cat([lightness, a_val, b_val], dim=1)

    def forward_structured(self, x: torch.Tensor) -> tuple:
        features = self.backbone(x)

        lightness = self.lightness_head(features) * 100.0

        hue_components = self.hue_head(features)
        hue_norm = torch.nn.functional.normalize(hue_components, dim=1)
        hue_angle = torch.atan2(hue_norm[:, 1:2], hue_norm[:, 0:1])

        chroma = self.chroma_head(features) * self.max_chroma

        return lightness, hue_angle, chroma
