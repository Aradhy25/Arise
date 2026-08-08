"""Unified inference engine for images and videos."""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from app.core.config import get_settings
from app.ml.architectures.factory import build_model, load_checkpoint
from app.ml.face_detector import FaceDetector
from app.ml.forensics import forensic_fake_probability, forensic_heatmap
from app.ml.gradcam import GradCAM, find_last_conv, overlay_heatmap
from app.ml.preprocessing import read_image, to_tensor
from app.ml.video import sample_video_frames


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


@dataclass
class InferenceResult:
    prediction: str
    confidence: float
    model_name: str
    model_version: str
    media_type: str
    frames_analyzed: int
    suspicious_frames: int
    processing_time_sec: float
    heatmap_path: str | None = None
    mode: str = "pytorch"
    details: dict = field(default_factory=dict)


class DeepfakeEngine:
    def __init__(self, model_name: str | None = None):
        self.settings = get_settings()
        self.model_name = model_name or self.settings.default_model
        self.device = torch.device(self.settings.device)
        self.face_detector = FaceDetector()
        self.model = build_model(self.model_name, pretrained=True)
        weights = self.settings.weights_dir / f"{self.model_name.replace('-', '_')}.pth"
        self.model, self.has_finetuned_weights = load_checkpoint(self.model, weights, self.device)
        self.model_version = "1.0-finetuned" if self.has_finetuned_weights else "1.0-baseline"

    def predict_file(self, path: str | Path, model_override: str | None = None) -> InferenceResult:
        path = Path(path)
        ext = path.suffix.lower()
        if model_override and model_override != self.model_name:
            engine = DeepfakeEngine(model_override)
            return engine.predict_file(path)

        if ext in IMAGE_EXTS:
            return self.predict_image(path)
        if ext in VIDEO_EXTS:
            return self.predict_video(path)
        raise ValueError(f"Unsupported file type: {ext}")

    def predict_image(self, path: Path) -> InferenceResult:
        t0 = time.perf_counter()
        image = read_image(str(path))
        face = self.face_detector.detect_or_full(image)
        face_img = face.image_rgb

        if self.has_finetuned_weights:
            fake_prob, mode, signals, heatmap = self._pytorch_predict(face_img)
        else:
            forensic = forensic_fake_probability(face_img)
            fake_prob = forensic["fake_probability"]
            mode = forensic["mode"]
            signals = forensic["signals"]
            heatmap = forensic_heatmap(face_img)
            # Blend a light model prior if desired (kept off to stay honest)

        prediction = "FAKE" if fake_prob >= self.settings.fake_threshold else "REAL"
        confidence = fake_prob if prediction == "FAKE" else 1.0 - fake_prob

        heatmap_path = self._save_heatmap(face_img, heatmap)
        elapsed = time.perf_counter() - t0

        return InferenceResult(
            prediction=prediction,
            confidence=round(float(confidence), 4),
            model_name=self.model_name,
            model_version=self.model_version,
            media_type="image",
            frames_analyzed=1,
            suspicious_frames=1 if prediction == "FAKE" else 0,
            processing_time_sec=round(elapsed, 3),
            heatmap_path=str(heatmap_path) if heatmap_path else None,
            mode=mode,
            details={
                "face_backend": self.face_detector.backend,
                "face_bbox": face.bbox,
                "face_confidence": face.confidence,
                "fake_probability": round(float(fake_prob), 4),
                "signals": signals,
                "finetuned_weights": self.has_finetuned_weights,
            },
        )

    def predict_video(self, path: Path) -> InferenceResult:
        t0 = time.perf_counter()
        sample = sample_video_frames(str(path), max_frames=self.settings.max_video_frames)
        probs: list[float] = []
        heatmaps: list[np.ndarray] = []
        faces_used = 0

        for frame in sample.frames:
            face = self.face_detector.detect_or_full(frame)
            face_img = face.image_rgb
            faces_used += 1 if face.confidence > 0 else 0

            if self.has_finetuned_weights:
                fake_prob, mode, signals, heat = self._pytorch_predict(face_img)
            else:
                forensic = forensic_fake_probability(face_img)
                fake_prob = forensic["fake_probability"]
                mode = forensic["mode"]
                signals = forensic["signals"]
                heat = forensic_heatmap(face_img)

            probs.append(fake_prob)
            heatmaps.append(heat)

        avg_prob = float(np.mean(probs)) if probs else 0.5
        suspicious = sum(1 for p in probs if p >= self.settings.fake_threshold)
        prediction = "FAKE" if avg_prob >= self.settings.fake_threshold else "REAL"
        confidence = avg_prob if prediction == "FAKE" else 1.0 - avg_prob

        # Save heatmap from most suspicious frame
        heatmap_path = None
        if heatmaps and sample.frames:
            idx = int(np.argmax(probs))
            face = self.face_detector.detect_or_full(sample.frames[idx])
            heatmap_path = self._save_heatmap(face.image_rgb, heatmaps[idx])

        elapsed = time.perf_counter() - t0
        return InferenceResult(
            prediction=prediction,
            confidence=round(float(confidence), 4),
            model_name=self.model_name,
            model_version=self.model_version,
            media_type="video",
            frames_analyzed=len(probs),
            suspicious_frames=suspicious,
            processing_time_sec=round(elapsed, 3),
            heatmap_path=str(heatmap_path) if heatmap_path else None,
            mode=mode if probs else "unknown",
            details={
                "face_backend": self.face_detector.backend,
                "faces_detected_frames": faces_used,
                "fake_probability": round(avg_prob, 4),
                "frame_probabilities": [round(p, 4) for p in probs],
                "video_total_frames": sample.total_frames,
                "video_fps": sample.fps,
                "video_duration_sec": round(sample.duration_sec, 2),
                "finetuned_weights": self.has_finetuned_weights,
                "aggregation": "mean_frame_probability",
            },
        )

    def _pytorch_predict(self, face_rgb: np.ndarray) -> tuple[float, str, dict, np.ndarray]:
        tensor = to_tensor(face_rgb, self.settings.image_size).unsqueeze(0).to(self.device)
        with torch.enable_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1)[0]
            fake_prob = float(probs[1].item())

            layer = find_last_conv(self.model)
            if layer is not None:
                cam_engine = GradCAM(self.model, layer)
                try:
                    cam = cam_engine.generate(tensor, class_idx=1 if fake_prob >= 0.5 else 0)
                finally:
                    cam_engine.close()
            else:
                cam = forensic_heatmap(face_rgb)

        signals = {
            "real_prob": round(float(probs[0].item()), 4),
            "fake_prob": round(fake_prob, 4),
        }
        return fake_prob, "pytorch", signals, cam

    def _save_heatmap(self, image_rgb: np.ndarray, cam: np.ndarray) -> Path | None:
        try:
            out_dir = self.settings.upload_dir / "heatmaps"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{uuid.uuid4().hex}_heatmap.jpg"
            overlay = overlay_heatmap(image_rgb, cam)
            cv2.imwrite(str(out_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            return out_path
        except Exception:
            return None


_engine: DeepfakeEngine | None = None


def get_engine() -> DeepfakeEngine:
    global _engine
    if _engine is None:
        _engine = DeepfakeEngine()
    return _engine
