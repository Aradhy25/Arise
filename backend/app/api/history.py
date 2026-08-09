"""Detection history routes."""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.api.detect import _to_out
from app.db.models import Detection, User
from app.db.session import get_db
from app.schemas import DetectionList, DetectionOut

router = APIRouter(prefix="/history", tags=["history"])


@router.get("", response_model=DetectionList)
def list_history(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> DetectionList:
    q = db.query(Detection).filter(Detection.user_id == user.id)
    total = q.count()
    rows = q.order_by(Detection.created_at.desc()).offset(offset).limit(limit).all()
    return DetectionList(items=[_to_out(r) for r in rows], total=total)


@router.get("/{detection_id}", response_model=DetectionOut)
def get_detection(
    detection_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> DetectionOut:
    det = (
        db.query(Detection)
        .filter(Detection.id == detection_id, Detection.user_id == user.id)
        .first()
    )
    if not det:
        raise HTTPException(status_code=404, detail="Detection not found")
    return _to_out(det)
