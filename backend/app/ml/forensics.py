"""Forensic heuristic analyzer used when no fine-tuned deepfake checkpoint exists.

Combines error-level analysis (ELA), frequency-domain energy, and local
noise inconsistency — cues commonly discussed in media forensics literature.
This keeps the demo pipeline honest: ImageNet-pretrained backbones alone are
not deepfake detectors.
"""

from __future__ import annotations

import io

import cv2
import numpy as np
from PIL import Image


def error_level_analysis(image_rgb: np.ndarray, quality: int = 90) -> float:
    """Higher ELA variance often indicates local manipulation / recompression."""
    pil = Image.fromarray(image_rgb)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    recompressed = np.array(Image.open(buf).convert("RGB"))
    diff = np.abs(image_rgb.astype(np.float32) - recompressed.astype(np.float32))
    return float(diff.std())


def frequency_anomaly_score(image_rgb: np.ndarray) -> float:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.log1p(np.abs(fshift))
    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    # Compare mid-frequency ring energy vs low-frequency energy
    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    low = magnitude[dist < min(h, w) * 0.1].mean()
    mid = magnitude[(dist >= min(h, w) * 0.15) & (dist < min(h, w) * 0.35)].mean()
    ratio = float(mid / (low + 1e-6))
    # Emphasize unusual mid-band energy
    return float(np.clip((ratio - 0.35) / 0.4, 0.0, 1.0))


def noise_inconsistency_score(image_rgb: np.ndarray) -> float:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    noise = gray - blur
    # Tile variance inconsistency
    h, w = noise.shape
    tile = 32
    vars_: list[float] = []
    for y in range(0, h - tile, tile):
        for x in range(0, w - tile, tile):
            patch = noise[y : y + tile, x : x + tile]
            vars_.append(float(patch.var()))
    if len(vars_) < 4:
        return 0.0
    arr = np.array(vars_)
    cv = float(arr.std() / (arr.mean() + 1e-6))
    return float(np.clip(cv / 1.5, 0.0, 1.0))


def color_channel_correlation_score(image_rgb: np.ndarray) -> float:
    """Deepfake blending can disrupt natural RGB channel correlation."""
    r, g, b = [image_rgb[:, :, i].astype(np.float32).ravel() for i in range(3)]
    rg = np.corrcoef(r, g)[0, 1]
    rb = np.corrcoef(r, b)[0, 1]
    gb = np.corrcoef(g, b)[0, 1]
    mean_corr = float(np.nanmean([rg, rb, gb]))
    # Natural faces typically have high correlation; low values are suspicious
    return float(np.clip((0.95 - mean_corr) / 0.35, 0.0, 1.0))


def forensic_fake_probability(image_rgb: np.ndarray) -> dict:
    ela = error_level_analysis(image_rgb)
    # Normalize ELA std (typical natural ~2–8, manipulated often higher)
    ela_score = float(np.clip((ela - 3.0) / 12.0, 0.0, 1.0))
    freq = frequency_anomaly_score(image_rgb)
    noise = noise_inconsistency_score(image_rgb)
    color = color_channel_correlation_score(image_rgb)

    # Weighted ensemble of forensic cues
    fake_prob = 0.35 * ela_score + 0.25 * freq + 0.25 * noise + 0.15 * color
    fake_prob = float(np.clip(fake_prob, 0.05, 0.95))

    return {
        "fake_probability": fake_prob,
        "signals": {
            "ela_score": round(ela_score, 4),
            "frequency_score": round(freq, 4),
            "noise_score": round(noise, 4),
            "color_score": round(color, 4),
            "ela_std": round(ela, 4),
        },
        "mode": "forensic_heuristic",
    }


def forensic_heatmap(image_rgb: np.ndarray) -> np.ndarray:
    """Simple ELA-based attention map for explainability in heuristic mode."""
    pil = Image.fromarray(image_rgb)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=90)
    buf.seek(0)
    recompressed = np.array(Image.open(buf).convert("RGB"))
    diff = np.abs(image_rgb.astype(np.float32) - recompressed.astype(np.float32)).mean(axis=2)
    diff = cv2.GaussianBlur(diff, (11, 11), 0)
    diff -= diff.min()
    if diff.max() > 0:
        diff /= diff.max()
    return diff.astype(np.float32)
