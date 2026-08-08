"""Detection / upload / live routes."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import aiofiles
import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.core.config import get_settings
from app.db.models import Detection, User
from app.db.session import get_db
from app.ml.inference import IMAGE_EXTS, VIDEO_EXTS, get_engine
from app.schemas import DetectionOut, LiveDetectionOut
from app.services.report import generate_report

router = APIRouter(prefix="/detect", tags=["detect"])


def _to_out(det: Detection) -> DetectionOut:
    heatmap_url = None
    report_url = None
    if det.heatmap_path:
        name = Path(det.heatmap_path).name
        heatmap_url = f"/files/heatmaps/{name}"
    if det.report_path:
        report_url = f"/files/reports/{Path(det.report_path).name}"
    details = json.loads(det.details_json) if det.details_json else None
    return DetectionOut(
        id=det.id,
        filename=det.filename,
        media_type=det.media_type,
        prediction=det.prediction,
        confidence=det.confidence,
        model_name=det.model_name,
        model_version=det.model_version,
        frames_analyzed=det.frames_analyzed,
        suspicious_frames=det.suspicious_frames,
        processing_time_sec=det.processing_time_sec,
        heatmap_url=heatmap_url,
        report_url=report_url,
        created_at=det.created_at,
        details=details,
    )


@router.post("", response_model=DetectionOut, status_code=status.HTTP_201_CREATED)
async def detect_media(
    file: UploadFile = File(...),
    model_name: str = Form(default="efficientnet"),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> DetectionOut:
    settings = get_settings()
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename required")

    ext = Path(file.filename).suffix.lower()
    if ext not in IMAGE_EXTS | VIDEO_EXTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {sorted(IMAGE_EXTS | VIDEO_EXTS)}",
        )

    content = await file.read()
    max_bytes = settings.max_upload_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(status_code=400, detail=f"File exceeds {settings.max_upload_mb}MB limit")

    media_type = "image" if ext in IMAGE_EXTS else "video"
    stored_name = f"{uuid.uuid4().hex}{ext}"
    dest = settings.upload_dir / stored_name
    async with aiofiles.open(dest, "wb") as f:
        await f.write(content)

    try:
        engine = get_engine()
        result = engine.predict_file(dest, model_override=model_name)
    except Exception as exc:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Inference failed: {exc}") from exc

    det = Detection(
        user_id=user.id,
        filename=file.filename,
        media_type=media_type,
        stored_path=str(dest),
        prediction=result.prediction,
        confidence=result.confidence,
        model_name=result.model_name,
        model_version=result.model_version,
        frames_analyzed=result.frames_analyzed,
        suspicious_frames=result.suspicious_frames,
        processing_time_sec=result.processing_time_sec,
        heatmap_path=result.heatmap_path,
        details_json=json.dumps(result.details),
    )
    db.add(det)
    db.commit()
    db.refresh(det)

    report_path = generate_report(
        detection_id=det.id,
        filename=det.filename,
        prediction=det.prediction,
        confidence=det.confidence,
        frames_analyzed=det.frames_analyzed,
        suspicious_frames=det.suspicious_frames,
        model_name=det.model_name,
        model_version=det.model_version,
        processing_time_sec=det.processing_time_sec,
        media_type=det.media_type,
    )
    det.report_path = str(report_path)
    db.commit()
    db.refresh(det)

    return _to_out(det)


@router.post("/live", response_model=LiveDetectionOut)
async def detect_live_frame(
    file: UploadFile = File(...),
    include_heatmap: bool = Form(default=True),
    user: User = Depends(get_current_user),
) -> LiveDetectionOut:
    """Real-time single-frame analysis for webcam streams (live PyTorch — no mock)."""
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty frame")
    if len(content) > 8 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Frame too large")

    arr = np.frombuffer(content, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode frame")

    try:
        result = get_engine().predict_frame_bgr(frame, include_heatmap=include_heatmap)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Live inference failed: {exc}") from exc

    bbox = result.details.get("face_bbox") if result.details else None
    return LiveDetectionOut(
        prediction=result.prediction,
        confidence=result.confidence,
        model_name=result.model_name,
        model_version=result.model_version,
        processing_time_sec=result.processing_time_sec,
        mode=result.mode,
        heatmap_b64=result.heatmap_b64,
        face_bbox=bbox,
        fake_probability=float(result.details.get("fake_probability", 0.0)),
        details=result.details,
    )


@router.get("/models")
def list_models() -> dict:
    engine = get_engine()
    return {
        "models": [
            {
                "id": "efficientnet",
                "name": "EfficientNet-B0",
                "phase": 1,
                "description": "Baseline CNN — live PyTorch inference + Grad-CAM",
            },
            {
                "id": "xception",
                "name": "Xception / ResNeXt-50",
                "phase": 2,
                "description": "Stronger CNN baseline",
            },
            {
                "id": "vit",
                "name": "Vision Transformer (ViT-B/16)",
                "phase": 3,
                "description": "Transformer-based research model",
            },
        ],
        "default": get_settings().default_model,
        "weights_loaded": engine.has_finetuned_weights,
        "inference_mode": "pytorch" if engine.has_finetuned_weights else "pytorch+forensics",
        "note": (
            "Every upload/live frame runs the real PyTorch model on the face crop. "
            "Place research checkpoints in backend/weights/ or run scripts/bootstrap_weights.py."
        ),
    }
