"""Model factory — EfficientNet, Xception, ViT."""

from __future__ import annotations

from pathlib import Path

import app.ml.compat  # noqa: F401  — lzma shim for pyenv Mac builds
import torch
import torch.nn as nn
from torchvision import models


class DeepfakeClassifier(nn.Module):
    """Binary real/fake classifier wrapping a backbone."""

    def __init__(self, backbone: nn.Module, feature_dim: int):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 2),
        )
        self.feature_dim = feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(x)
        return self.classifier(features)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def _efficientnet(variant: str = "b0", pretrained: bool = True) -> DeepfakeClassifier:
    weights = "DEFAULT" if pretrained else None
    if variant == "b2":
        net = models.efficientnet_b2(weights=weights)
        feature_dim = net.classifier[1].in_features
    else:
        net = models.efficientnet_b0(weights=weights)
        feature_dim = net.classifier[1].in_features
    net.classifier = nn.Identity()
    return DeepfakeClassifier(net, feature_dim)


def _xception_like(pretrained: bool = True) -> DeepfakeClassifier:
    """Xception-style backbone via ResNeXt50 (torchvision-native stand-in).

    Full Xception requires timm; ResNeXt50 provides a strong CNN baseline
    with similar capacity for academic comparison until timm is installed.
    """
    weights = "DEFAULT" if pretrained else None
    net = models.resnext50_32x4d(weights=weights)
    feature_dim = net.fc.in_features
    net.fc = nn.Identity()
    return DeepfakeClassifier(net, feature_dim)


def _vit(pretrained: bool = True) -> DeepfakeClassifier:
    weights = "DEFAULT" if pretrained else None
    net = models.vit_b_16(weights=weights)
    feature_dim = net.heads.head.in_features
    net.heads.head = nn.Identity()
    return DeepfakeClassifier(net, feature_dim)


def build_model(name: str = "efficientnet", pretrained: bool = True) -> DeepfakeClassifier:
    key = name.lower().strip()
    if key in {"efficientnet", "efficientnet-b0", "efficientnet_b0"}:
        return _efficientnet("b0", pretrained=pretrained)
    if key in {"efficientnet-b2", "efficientnet_b2"}:
        return _efficientnet("b2", pretrained=pretrained)
    if key in {"xception", "xceptionnet", "resnext"}:
        return _xception_like(pretrained=pretrained)
    if key in {"vit", "vision_transformer", "vit_b_16"}:
        return _vit(pretrained=pretrained)
    raise ValueError(f"Unknown model: {name}")


def load_checkpoint(
    model: nn.Module,
    weights_path: Path | None,
    device: torch.device,
) -> tuple[nn.Module, bool]:
    """Load fine-tuned weights if present. Returns (model, loaded_flag)."""
    if weights_path and weights_path.exists():
        state = torch.load(weights_path, map_location=device, weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=False)
        model.to(device)
        model.eval()
        return model, True
    model.to(device)
    model.eval()
    return model, False
