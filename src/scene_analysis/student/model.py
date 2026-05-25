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
    def __init__(self, in_channels: int, out_channels: int, dropout: float) -> None:
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout2d(p=dropout))
        super().__init__(*layers)


class FpnDecoder(nn.Module):
    def __init__(self, feature_channels: Sequence[int], decoder_channels: Sequence[int], dropout: float) -> None:
        super().__init__()
        channels = [int(item) for item in decoder_channels]
        if not channels:
            raise ValueError("decoder_channels must not be empty")

        fpn_channels = channels[0]
        self.lateral_convs = nn.ModuleList(
            nn.Conv2d(in_channels, fpn_channels, kernel_size=1)
            for in_channels in feature_channels
        )
        self.output_convs = nn.ModuleList(
            ConvBnRelu(fpn_channels, fpn_channels, dropout=0.0)
            for _ in feature_channels
        )

        fusion_channels = [fpn_channels * len(feature_channels), *channels]
        self.fusion = nn.Sequential(
            *[
                ConvBnRelu(in_channels, out_channels, dropout=dropout)
                for in_channels, out_channels in zip(fusion_channels[:-1], fusion_channels[1:])
            ]
        )
        self.out_channels = channels[-1]

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(features) != len(self.lateral_convs):
            raise ValueError(f"Expected {len(self.lateral_convs)} feature maps, got {len(features)}")

        pyramid: list[torch.Tensor] = [self.lateral_convs[-1](features[-1])]
        for feature, lateral_conv in zip(reversed(features[:-1]), reversed(self.lateral_convs[:-1])):
            top_down = F.interpolate(
                pyramid[-1],
                size=feature.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            pyramid.append(lateral_conv(feature) + top_down)
        pyramid = list(reversed(pyramid))

        output_size = pyramid[0].shape[-2:]
        fused_levels = [
            output_conv(level)
            for output_conv, level in zip(self.output_convs, pyramid)
        ]
        fused_levels = [
            level
            if level.shape[-2:] == output_size
            else F.interpolate(level, size=output_size, mode="bilinear", align_corners=False)
            for level in fused_levels
        ]
        return self.fusion(torch.cat(fused_levels, dim=1))


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
        self.backbone, feature_channels = _create_backbone(backbone_name, pretrained_backbone)

        self.decoder = FpnDecoder(feature_channels, decoder_channels, dropout=dropout)
        self.obstacle_head = nn.Conv2d(self.decoder.out_channels, 1, kernel_size=1)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        input_size = image.shape[-2:]
        features = self.backbone(image)
        x = self.decoder(features)
        if x.shape[-2:] != input_size:
            x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)

        obstacle_logits = self.obstacle_head(x)
        obstacle_prob = torch.sigmoid(obstacle_logits)
        return {
            "obstacle_logits": obstacle_logits,
            "obstacle_prob": obstacle_prob,
            "final_heatmap": obstacle_prob,
        }


class MobileNetV3SmallFeatures(nn.Module):
    channels = [16, 24, 48, 576]

    def __init__(self, pretrained: bool) -> None:
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        self.features = mobilenet_v3_small(weights=weights).features
        self.return_indices = {1, 3, 8, 12}

    def forward(self, image: torch.Tensor) -> list[torch.Tensor]:
        outputs: list[torch.Tensor] = []
        x = image
        for index, layer in enumerate(self.features):
            x = layer(x)
            if index in self.return_indices:
                outputs.append(x)
        return outputs


class ShuffleNetV2Features(nn.Module):
    channels = [24, 116, 232, 1024]

    def __init__(self, pretrained: bool) -> None:
        super().__init__()
        weights = ShuffleNet_V2_X1_0_Weights.DEFAULT if pretrained else None
        model = shufflenet_v2_x1_0(weights=weights)
        self.conv1 = model.conv1
        self.maxpool = model.maxpool
        self.stage2 = model.stage2
        self.stage3 = model.stage3
        self.stage4 = model.stage4
        self.conv5 = model.conv5

    def forward(self, image: torch.Tensor) -> list[torch.Tensor]:
        x = self.conv1(image)
        c2 = self.maxpool(x)
        c3 = self.stage2(c2)
        c4 = self.stage3(c3)
        x = self.stage4(c4)
        c5 = self.conv5(x)
        return [c2, c3, c4, c5]


class EfficientNetB0Features(nn.Module):
    channels = [24, 40, 112, 1280]

    def __init__(self, pretrained: bool) -> None:
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.features = efficientnet_b0(weights=weights).features
        self.return_indices = {2, 3, 5, 8}

    def forward(self, image: torch.Tensor) -> list[torch.Tensor]:
        outputs: list[torch.Tensor] = []
        x = image
        for index, layer in enumerate(self.features):
            x = layer(x)
            if index in self.return_indices:
                outputs.append(x)
        return outputs


def count_parameters(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def _create_backbone(backbone_name: str, pretrained: bool) -> tuple[nn.Module, list[int]]:
    normalized = backbone_name.strip().lower()
    if normalized == "mobilenet_v3_small":
        backbone = MobileNetV3SmallFeatures(pretrained)
        return backbone, backbone.channels
    if normalized == "shufflenet_v2_x1_0":
        backbone = ShuffleNetV2Features(pretrained)
        return backbone, backbone.channels
    if normalized == "efficientnet_b0":
        backbone = EfficientNetB0Features(pretrained)
        return backbone, backbone.channels
    raise ValueError(f"Unsupported student backbone: {backbone_name}")
