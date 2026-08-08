"""Grad-CAM explainability for CNN backbones."""

from __future__ import annotations

from typing import Callable

import cv2
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    """Compute Grad-CAM heatmaps for the last convolutional layer."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._handles = [
            target_layer.register_forward_hook(self._forward_hook),
            target_layer.register_full_backward_hook(self._backward_hook),
        ]

    def _forward_hook(self, _module, _inp, output) -> None:
        self.activations = output.detach()

    def _backward_hook(self, _module, _grad_input, grad_output) -> None:
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(input_tensor)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())
        score = logits[0, class_idx]
        score.backward()

        assert self.gradients is not None and self.activations is not None
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam

    def close(self) -> None:
        for h in self._handles:
            h.remove()


def find_last_conv(module: torch.nn.Module) -> torch.nn.Module | None:
    last = None
    for m in module.modules():
        if isinstance(m, torch.nn.Conv2d):
            last = m
    return last


def overlay_heatmap(image_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    heat = cv2.resize(cam, (image_rgb.shape[1], image_rgb.shape[0]))
    heat_uint8 = np.uint8(255 * heat)
    colored = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    base = image_rgb.astype(np.float32)
    blended = (1 - alpha) * base + alpha * colored.astype(np.float32)
    return np.clip(blended, 0, 255).astype(np.uint8)


def save_heatmap(path: str, image_rgb: np.ndarray, cam: np.ndarray) -> None:
    overlay = overlay_heatmap(image_rgb, cam)
    cv2.imwrite(path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
