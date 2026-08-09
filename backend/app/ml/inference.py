"""Unified inference engine for images, videos, and live frames."""

from __future__ import annotations

import base64
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from app.core.config import get_settings
from app.ml.audio import AUDIO_EXTS, analyze_audio
from app.ml.architectures.factory import build_model, load_checkpoint
from app.ml.face_detector import FaceDetector
from app.ml.forensics import forensic_fake_probability, forensic_heatmap
from app.ml.gradcam import GradCAM, find_last_conv, overlay_heatmap
from app.ml.preprocessing import read_image, to_tensor
from app.ml.risk import explain_result, file_sha256, risk_level
from app.ml.video import sample_video_frames


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
ALL_MEDIA_EXTS = IMAGE_EXTS | VIDEO_EXTS | AUDIO_EXTS


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
    heatmap_b64: str | None = None
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
        if self.has_finetuned_weights:
            self.model_version = "1.0-finetuned"
        else:
            # Still run the network — but mark that research datasets should replace bootstrap
            self.model_version = "1.0-imagenet-head"

    def predict_file(self, path: str | Path, model_override: str | None = None) -> InferenceResult:
        path = Path(path)
        ext = path.suffix.lower()
        if model_override and model_override != self.model_name:
            return DeepfakeEngine(model_override).predict_file(path)

        if ext in IMAGE_EXTS:
            result = self.predict_image(path)
        elif ext in VIDEO_EXTS:
            result = self.predict_video(path)
        elif ext in AUDIO_EXTS:
            result = self.predict_audio(path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
        return enrich_result(result, path)

    def predict_ensemble(self, path: str | Path, models: list[str] | None = None) -> InferenceResult:
        """Run multiple visual models and vote (majority + mean probability)."""
        path = Path(path)
        ext = path.suffix.lower()
        if ext in AUDIO_EXTS:
            return self.predict_file(path)

        models = models or ["efficientnet", "xception", "vit"]
        t0 = time.perf_counter()
        votes: list[InferenceResult] = []
        for name in models:
            try:
                votes.append(DeepfakeEngine(name).predict_file(path))
            except Exception:
                continue
        if not votes:
            raise ValueError("Ensemble failed — no model produced a result")

        probs = [float(v.details.get("fake_probability", 0.5)) for v in votes]
        avg = float(np.mean(probs))
        fake_votes = sum(1 for v in votes if v.prediction == "FAKE")
        prediction = "FAKE" if fake_votes > len(votes) / 2 or avg >= self.settings.fake_threshold else "REAL"
        confidence = avg if prediction == "FAKE" else 1.0 - avg
        best = max(votes, key=lambda v: float(v.details.get("fake_probability", 0)))

        result = InferenceResult(
            prediction=prediction,
            confidence=round(float(confidence), 4),
            model_name="ensemble",
            model_version="1.0",
            media_type=votes[0].media_type,
            frames_analyzed=votes[0].frames_analyzed,
            suspicious_frames=votes[0].suspicious_frames,
            processing_time_sec=round(time.perf_counter() - t0, 3),
            heatmap_path=best.heatmap_path,
            mode="ensemble",
            details={
                "fake_probability": round(avg, 4),
                "ensemble": [
                    {
                        "model": v.model_name,
                        "prediction": v.prediction,
                        "confidence": v.confidence,
                        "fake_probability": v.details.get("fake_probability"),
                    }
                    for v in votes
                ],
                "votes_fake": fake_votes,
                "votes_total": len(votes),
                "frame_probabilities": votes[0].details.get("frame_probabilities"),
                "signals": votes[0].details.get("signals"),
                "aggregation": "majority_vote+mean_probability",
            },
        )
        return enrich_result(result, path)

    def predict_audio(self, path: Path) -> InferenceResult:
        t0 = time.perf_counter()
        result = analyze_audio(path)
        elapsed = time.perf_counter() - t0
        return InferenceResult(
            prediction=result["prediction"],
            confidence=result["confidence"],
            model_name="audio-forensics",
            model_version="1.0",
            media_type="audio",
            frames_analyzed=1,
            suspicious_frames=1 if result["prediction"] == "FAKE" else 0,
            processing_time_sec=round(elapsed, 3),
            mode=result["mode"],
            details={
                "fake_probability": result["fake_probability"],
                "signals": result["signals"],
                "finetuned_weights": False,
                "modality": "audio",
            },
        )

    def predict_image(self, path: Path) -> InferenceResult:
        t0 = time.perf_counter()
        image = read_image(str(path))
        face = self.face_detector.detect_or_full(image)
        fake_prob, mode, signals, heatmap = self._predict_face(face.image_rgb)

        prediction = "FAKE" if fake_prob >= self.settings.fake_threshold else "REAL"
        confidence = fake_prob if prediction == "FAKE" else 1.0 - fake_prob
        heatmap_path = self._save_heatmap(face.image_rgb, heatmap)
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
                "face_bbox": list(face.bbox),
                "face_confidence": float(face.confidence),
                "fake_probability": round(float(fake_prob), 4),
                "signals": signals,
                "finetuned_weights": self.has_finetuned_weights,
                "realtime": False,
            },
        )

    def predict_video(self, path: Path) -> InferenceResult:
        t0 = time.perf_counter()
        sample = sample_video_frames(str(path), max_frames=self.settings.max_video_frames)
        probs: list[float] = []
        heatmaps: list[np.ndarray] = []
        faces_used = 0
        mode = "pytorch"
        last_signals: dict = {}

        for frame in sample.frames:
            face = self.face_detector.detect_or_full(frame)
            if face.confidence > 0:
                faces_used += 1
            fake_prob, mode, last_signals, heat = self._predict_face(face.image_rgb)
            probs.append(fake_prob)
            heatmaps.append(heat)

        avg_prob = float(np.mean(probs)) if probs else 0.5
        suspicious = sum(1 for p in probs if p >= self.settings.fake_threshold)
        prediction = "FAKE" if avg_prob >= self.settings.fake_threshold else "REAL"
        confidence = avg_prob if prediction == "FAKE" else 1.0 - avg_prob

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
            mode=mode,
            details={
                "face_backend": self.face_detector.backend,
                "faces_detected_frames": faces_used,
                "fake_probability": round(avg_prob, 4),
                "frame_probabilities": [round(p, 4) for p in probs],
                "video_total_frames": sample.total_frames,
                "video_fps": sample.fps,
                "video_duration_sec": round(sample.duration_sec, 2),
                "finetuned_weights": self.has_finetuned_weights,
                "signals": last_signals,
                "aggregation": "mean_frame_probability",
                "realtime": False,
            },
        )

    def predict_frame_bgr(self, frame_bgr: np.ndarray, *, include_heatmap: bool = True) -> InferenceResult:
        """Fast path for live webcam frames (numpy BGR from OpenCV / decoded JPEG)."""
        t0 = time.perf_counter()
        image = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        face = self.face_detector.detect_or_full(image)
        fake_prob, mode, signals, heatmap = self._predict_face(face.image_rgb)
        prediction = "FAKE" if fake_prob >= self.settings.fake_threshold else "REAL"
        confidence = fake_prob if prediction == "FAKE" else 1.0 - fake_prob

        heatmap_b64 = None
        heatmap_path = None
        if include_heatmap:
            overlay = overlay_heatmap(face.image_rgb, heatmap)
            ok, buf = cv2.imencode(".jpg", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ok:
                heatmap_b64 = base64.b64encode(buf.tobytes()).decode("ascii")

        elapsed = time.perf_counter() - t0
        result = InferenceResult(
            prediction=prediction,
            confidence=round(float(confidence), 4),
            model_name=self.model_name,
            model_version=self.model_version,
            media_type="live",
            frames_analyzed=1,
            suspicious_frames=1 if prediction == "FAKE" else 0,
            processing_time_sec=round(elapsed, 3),
            heatmap_path=heatmap_path,
            heatmap_b64=heatmap_b64,
            mode=mode,
            details={
                "face_backend": self.face_detector.backend,
                "face_bbox": list(face.bbox),
                "face_confidence": float(face.confidence),
                "fake_probability": round(float(fake_prob), 4),
                "signals": signals,
                "finetuned_weights": self.has_finetuned_weights,
                "realtime": True,
            },
        )
        return enrich_result(result)

    def _predict_face(self, face_rgb: np.ndarray) -> tuple[float, str, dict, np.ndarray]:
        """Always run the PyTorch model + Grad-CAM on the face crop."""
        tensor = to_tensor(face_rgb, self.settings.image_size).unsqueeze(0).to(self.device)
        with torch.enable_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1)[0]
            model_fake = float(probs[1].item())

            layer = find_last_conv(self.model)
            if layer is not None:
                cam_engine = GradCAM(self.model, layer)
                try:
                    cam = cam_engine.generate(tensor, class_idx=1 if model_fake >= 0.5 else 0)
                finally:
                    cam_engine.close()
            else:
                cam = forensic_heatmap(face_rgb)

        # Auxiliary forensic signals (real analysis of this frame — not mock labels)
        forensic = forensic_fake_probability(face_rgb)
        forensic_p = forensic["fake_probability"]

        # If we have fine-tuned weights, trust the network; else fuse lightly with forensics
        if self.has_finetuned_weights:
            fake_prob = model_fake
            mode = "pytorch"
        else:
            fake_prob = 0.7 * model_fake + 0.3 * forensic_p
            mode = "pytorch+forensics"

        signals = {
            "model_fake_prob": round(model_fake, 4),
            "model_real_prob": round(float(probs[0].item()), 4),
            "forensic_fake_prob": round(forensic_p, 4),
            **{f"forensic_{k}": v for k, v in forensic["signals"].items()},
        }
        return float(np.clip(fake_prob, 0.0, 1.0)), mode, signals, cam

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


def enrich_result(result: InferenceResult, path: Path | None = None) -> InferenceResult:
    """Attach risk tier, SHA-256, and plain-language explanation."""
    fake_p = float(result.details.get("fake_probability", result.confidence if result.prediction == "FAKE" else 1 - result.confidence))
    ratio = result.suspicious_frames / max(result.frames_analyzed, 1)
    risk = risk_level(fake_p, ratio)
    explanation = explain_result(
        prediction=result.prediction,
        fake_probability=fake_p,
        media_type=result.media_type,
        mode=result.mode,
        signals=result.details.get("signals") if isinstance(result.details.get("signals"), dict) else None,
        frames_analyzed=result.frames_analyzed,
        suspicious_frames=result.suspicious_frames,
    )
    result.details = {
        **result.details,
        "risk": risk,
        "explanation": explanation,
        "sha256": file_sha256(path) if path and Path(path).exists() else None,
    }
    return result


def get_engine() -> DeepfakeEngine:
    global _engine
    if _engine is None:
        _engine = DeepfakeEngine()
    return _engine


def reload_engine() -> DeepfakeEngine:
    global _engine
    _engine = DeepfakeEngine()
    return _engine
