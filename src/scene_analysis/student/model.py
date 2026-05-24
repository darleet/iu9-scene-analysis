from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import (
    EfficientNet_B0_Weights,
    MobileNet_V3_Small_Weights,
    ShuffleNet_V2_X1_0_Weights,
    efficientnet_b0,
    mobilenet_v3_small,
    shufflenet_v2_x1_0,
)


class ConvBnRelu(nn.Sequential):
    """Декодер для student моделей"""

    def __init__(self, in_channels: int, out_channels: int, dropout: float) -> None:
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout2d(p=dropout))
        super().__init__(*layers)


class StudentHeatmapNet(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        pretrained_backbone: bool,
        decoder_channels: Sequence[int],
        dropout: float,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.pretrained_backbone = pretrained_backbone
        self.backbone, backbone_channels = _create_backbone(backbone_name, pretrained_backbone)

        channels = [backbone_channels, *[int(item) for item in decoder_channels]]
        decoder_layers: list[nn.Module] = []
        for in_channels, out_channels in zip(channels[:-1], channels[1:]):
            decoder_layers.append(ConvBnRelu(in_channels, out_channels, dropout=dropout))
        self.decoder = nn.ModuleList(decoder_layers)
        head_channels = channels[-1]
        self.obstacle_head = nn.Conv2d(head_channels, 1, kernel_size=1)
        self.roi_head = nn.Conv2d(head_channels, 1, kernel_size=1)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        input_size = image.shape[-2:]
        features = self.backbone(image)
        x = features
        for block in self.decoder:
            x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
            x = block(x)
        if x.shape[-2:] != input_size:
            x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)

        obstacle_logits = self.obstacle_head(x)
        roi_logits = self.roi_head(x)
        obstacle_prob = torch.sigmoid(obstacle_logits)
        roi_prob = torch.sigmoid(roi_logits)
        final_heatmap = obstacle_prob * roi_prob
        return {
            "obstacle_logits": obstacle_logits,
            "roi_logits": roi_logits,
            "obstacle_prob": obstacle_prob,
            "roi_prob": roi_prob,
            "final_heatmap": final_heatmap,
        }


def count_parameters(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def _create_backbone(backbone_name: str, pretrained: bool) -> tuple[nn.Module, int]:
    normalized = backbone_name.strip().lower()
    if normalized == "mobilenet_v3_small":
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = mobilenet_v3_small(weights=weights)
        return model.features, 576
    if normalized == "shufflenet_v2_x1_0":
        weights = ShuffleNet_V2_X1_0_Weights.DEFAULT if pretrained else None
        model = shufflenet_v2_x1_0(weights=weights)
        features = nn.Sequential(model.conv1, model.maxpool, model.stage2, model.stage3, model.stage4, model.conv5)
        return features, 1024
    if normalized == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = efficientnet_b0(weights=weights)
        return model.features, 1280
    raise ValueError(f"Unsupported student backbone: {backbone_name}")
