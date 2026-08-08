"""API and inference tests for DeepGuard AI."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

# Ensure backend package is importable
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.config import get_settings
from app.db.session import Base, engine
from app.main import create_app
from app.ml.forensics import forensic_fake_probability
from app.ml.face_detector import FaceDetector
from app.ml.video import _uniform_indices


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    get_settings.cache_clear()
    settings = get_settings()
    tmp = tmp_path_factory.mktemp("deepguard")
    settings.upload_dir = tmp / "uploads"
    settings.reports_dir = tmp / "reports"
    settings.weights_dir = tmp / "weights"
    settings.upload_dir.mkdir(parents=True)
    settings.reports_dir.mkdir(parents=True)
    settings.weights_dir.mkdir(parents=True)
    settings.database_url = f"sqlite:///{tmp / 'test.db'}"

    # Recreate engine tables for sqlite test db — use app's Base on fresh engine
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    import app.db.session as db_session

    test_engine = create_engine(settings.database_url, connect_args={"check_same_thread": False})
    TestingSession = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
    Base.metadata.create_all(bind=test_engine)

    db_session.engine = test_engine
    db_session.SessionLocal = TestingSession

    app = create_app()

    def override_get_db():
        db = TestingSession()
        try:
            yield db
        finally:
            db.close()

    from app.db.session import get_db

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as c:
        yield c

    get_settings.cache_clear()


def _make_image_bytes(color=(180, 140, 120), size=(256, 256)) -> bytes:
    img = Image.new("RGB", size, color)
    # Add mild noise so forensic signals are non-trivial
    arr = np.array(img).astype(np.int16)
    noise = np.random.randint(-8, 9, arr.shape, dtype=np.int16)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=92)
    return buf.getvalue()


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    data = r.json()
    assert data["app"] == "DeepGuard AI"
    assert data["status"] in {"ok", "degraded"}


def test_register_login_me(client):
    r = client.post(
        "/api/auth/register",
        json={"email": "alice@example.com", "full_name": "Alice", "password": "secret12"},
    )
    assert r.status_code == 201
    token = r.json()["access_token"]
    assert r.json()["user"]["role"] == "admin"  # first user

    r2 = client.post(
        "/api/auth/login",
        json={"email": "alice@example.com", "password": "secret12"},
    )
    assert r2.status_code == 200

    me = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 200
    assert me.json()["email"] == "alice@example.com"


def test_detect_image_and_history(client):
    # Ensure auth
    reg = client.post(
        "/api/auth/register",
        json={"email": "bob@example.com", "full_name": "Bob", "password": "secret12"},
    )
    if reg.status_code == 201:
        token = reg.json()["access_token"]
    else:
        login = client.post(
            "/api/auth/login",
            json={"email": "alice@example.com", "password": "secret12"},
        )
        token = login.json()["access_token"]

    headers = {"Authorization": f"Bearer {token}"}
    files = {"file": ("sample.jpg", _make_image_bytes(), "image/jpeg")}
    data = {"model_name": "efficientnet"}
    r = client.post("/api/detect", headers=headers, files=files, data=data)
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["prediction"] in {"REAL", "FAKE"}
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["frames_analyzed"] == 1
    assert body["report_url"]

    hist = client.get("/api/history", headers=headers)
    assert hist.status_code == 200
    assert hist.json()["total"] >= 1


def test_reject_invalid_file(client):
    login = client.post(
        "/api/auth/login",
        json={"email": "alice@example.com", "password": "secret12"},
    )
    if login.status_code != 200:
        login = client.post(
            "/api/auth/register",
            json={"email": "carol@example.com", "full_name": "Carol", "password": "secret12"},
        )
    token = login.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    files = {"file": ("notes.txt", b"not media", "text/plain")}
    r = client.post("/api/detect", headers=headers, files=files, data={"model_name": "efficientnet"})
    assert r.status_code == 400


def test_forensics_runs():
    img = np.random.randint(40, 220, (224, 224, 3), dtype=np.uint8)
    result = forensic_fake_probability(img)
    assert 0.0 <= result["fake_probability"] <= 1.0
    assert "signals" in result


def test_face_detector_fallback():
    det = FaceDetector()
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    # Draw a simple bright square "face-like" region — haar may or may not fire;
    # detect_or_full must always return something.
    crop = det.detect_or_full(img)
    assert crop.image_rgb.size > 0


def test_uniform_indices():
    assert _uniform_indices(100, 5) == [0, 24, 49, 74, 99]
    assert _uniform_indices(1, 10) == [0]


def test_live_detect(client):
    login = client.post(
        "/api/auth/login",
        json={"email": "alice@example.com", "password": "secret12"},
    )
    if login.status_code != 200:
        login = client.post(
            "/api/auth/register",
            json={"email": "live@example.com", "full_name": "Live", "password": "secret12"},
        )
    token = login.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    files = {"file": ("frame.jpg", _make_image_bytes(), "image/jpeg")}
    r = client.post("/api/detect/live", headers=headers, files=files, data={"include_heatmap": "true"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["prediction"] in {"REAL", "FAKE"}
    assert body["mode"] in {"pytorch", "pytorch+forensics"}
    assert body["heatmap_b64"]
    assert body["details"]["signals"]["model_fake_prob"] is not None


def test_models_endpoint(client):
    r = client.get("/api/detect/models")
    assert r.status_code == 200
    assert len(r.json()["models"]) == 3
    assert "inference_mode" in r.json()
