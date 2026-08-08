"""Application settings for DeepGuard AI."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "DeepGuard AI"
    app_version: str = "1.0.0"
    debug: bool = True

    # Auth
    secret_key: str = "deepguard-dev-secret-change-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60 * 24

    # Database — SQLite by default; set DATABASE_URL for PostgreSQL
    database_url: str = "sqlite:///./deepguard.db"

    # Storage
    upload_dir: Path = Path(__file__).resolve().parents[2] / "uploads"
    weights_dir: Path = Path(__file__).resolve().parents[2] / "weights"
    reports_dir: Path = Path(__file__).resolve().parents[2] / "reports"
    max_upload_mb: int = 100

    # Inference
    default_model: str = "efficientnet"
    image_size: int = 224
    max_video_frames: int = 40
    fake_threshold: float = 0.5
    device: str = "cpu"

    # CORS
    cors_origins: str = "http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173"


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.weights_dir.mkdir(parents=True, exist_ok=True)
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    return settings
