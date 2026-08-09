"""Pydantic schemas."""

from datetime import datetime

from pydantic import BaseModel, EmailStr, Field


class UserCreate(BaseModel):
    email: EmailStr
    full_name: str = Field(min_length=2, max_length=255)
    password: str = Field(min_length=6, max_length=128)


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserOut(BaseModel):
    id: int
    email: EmailStr
    full_name: str
    role: str
    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut


class DetectionOut(BaseModel):
    id: int
    filename: str
    media_type: str
    prediction: str
    confidence: float
    model_name: str
    model_version: str
    frames_analyzed: int
    suspicious_frames: int
    processing_time_sec: float
    heatmap_url: str | None = None
    report_url: str | None = None
    created_at: datetime
    details: dict | None = None

    model_config = {"from_attributes": True}


class LiveDetectionOut(BaseModel):
    prediction: str
    confidence: float
    model_name: str
    model_version: str
    processing_time_sec: float
    mode: str
    heatmap_b64: str | None = None
    face_bbox: list[int] | None = None
    fake_probability: float
    details: dict | None = None


class PublicDetectionOut(BaseModel):
    """Guest / public scan result (no account required)."""

    prediction: str
    confidence: float
    model_name: str
    model_version: str
    media_type: str
    frames_analyzed: int
    suspicious_frames: int
    processing_time_sec: float
    mode: str
    heatmap_url: str | None = None
    fake_probability: float
    details: dict | None = None
    guest: bool = True
    risk_level: str | None = None
    risk_label: str | None = None
    explanation: list[str] | None = None
    sha256: str | None = None


class BatchDetectionOut(BaseModel):
    items: list[PublicDetectionOut]
    total: int
    fake_count: int
    real_count: int


class AdminStatsOut(BaseModel):
    total_detections: int
    fake_count: int
    real_count: int
    image_count: int
    video_count: int
    audio_count: int
    users: int
    avg_confidence: float


class DetectionList(BaseModel):
    items: list[DetectionOut]
    total: int


class HealthOut(BaseModel):
    status: str
    app: str
    version: str
    model: str
    device: str
    weights_loaded: bool = False
    inference_mode: str = "pytorch"
