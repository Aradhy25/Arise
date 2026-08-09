"""Risk scoring and human-readable forensic explanations."""

from __future__ import annotations

import hashlib
from pathlib import Path


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def risk_level(fake_probability: float, suspicious_ratio: float = 0.0) -> dict:
    """Map probabilities to actionable risk tiers."""
    score = max(float(fake_probability), float(suspicious_ratio) * 0.85)
    if score >= 0.85:
        level, label, color = "critical", "Critical — strong manipulation evidence", "#7A271A"
    elif score >= 0.65:
        level, label, color = "high", "High — likely synthetic / manipulated", "#B42318"
    elif score >= 0.45:
        level, label, color = "medium", "Medium — mixed / uncertain signals", "#B54708"
    elif score >= 0.25:
        level, label, color = "low", "Low — mostly consistent with authentic media", "#027A48"
    else:
        level, label, color = "minimal", "Minimal — no strong forgery cues", "#027A48"
    return {
        "level": level,
        "label": label,
        "color": color,
        "score": round(score, 4),
    }


def explain_result(
    *,
    prediction: str,
    fake_probability: float,
    media_type: str,
    mode: str,
    signals: dict | None = None,
    frames_analyzed: int = 1,
    suspicious_frames: int = 0,
) -> list[str]:
    """Plain-language bullets for the UI / report."""
    lines: list[str] = []
    pct = round(fake_probability * 100, 1)
    if prediction == "FAKE":
        lines.append(f"Model estimates a {pct}% chance this {media_type} is synthetically manipulated.")
    else:
        lines.append(f"Model estimates a {pct}% chance of manipulation; overall verdict is authentic-leaning.")

    if media_type == "video" and frames_analyzed:
        ratio = suspicious_frames / max(frames_analyzed, 1)
        lines.append(
            f"Analyzed {frames_analyzed} frames; {suspicious_frames} flagged "
            f"({ratio * 100:.0f}% suspicious frame rate)."
        )
    if media_type == "audio" and signals:
        lines.append(
            "Audio forensics checked spectral flatness, high-frequency energy, "
            "and zero-crossing stability (common TTS / voice-clone cues)."
        )
    if mode.startswith("pytorch"):
        lines.append("Visual path used a PyTorch classifier with Grad-CAM attention mapping.")
    if signals:
        if "forensic_ela_score" in signals or "ela_score" in signals:
            lines.append("Error-level / compression inconsistency cues were included in the score.")
        if "model_fake_prob" in signals:
            lines.append(f"Neural net fake probability: {round(float(signals['model_fake_prob']) * 100, 1)}%.")
    lines.append("Treat this as forensic decision-support — not absolute legal proof.")
    return lines
