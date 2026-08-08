"""Detection / upload routes."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import aiofiles
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.core.config import get_settings
from app.db.models import Detection, User
from app.db.session import get_db
from app.ml.inference import IMAGE_EXTS, VIDEO_EXTS, get_engine
from app.schemas import DetectionOut
from app.services.report import generate_report

router = APIRouter(prefix="/detect", tags=["detect"])


def _to_out(det: Detection) -> DetectionOut:
    settings = get_settings()
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


@router.get("/models")
def list_models() -> dict:
    return {
        "models": [
            {
                "id": "efficientnet",
                "name": "EfficientNet-B0",
                "phase": 1,
                "description": "Baseline CNN — fast and strong for academic comparison",
            },
            {
                "id": "xception",
                "name": "Xception / ResNeXt-50",
                "phase": 2,
                "description": "Stronger CNN baseline (ResNeXt stand-in; swap for timm Xception)",
            },
            {
                "id": "vit",
                "name": "Vision Transformer (ViT-B/16)",
                "phase": 3,
                "description": "Transformer-based research model",
            },
        ],
        "default": get_settings().default_model,
        "note": (
            "Without fine-tuned deepfake weights, inference uses forensic heuristics "
            "(ELA / frequency / noise). Place *.pth checkpoints in backend/weights/ "
            "to enable PyTorch Grad-CAM mode."
        ),
    }
