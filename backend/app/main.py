"""DeepGuard AI — FastAPI application entrypoint."""

from contextlib import asynccontextmanager
from pathlib import Path

import app.ml.compat  # noqa: F401  — must run before torchvision imports

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api import auth, detect, history
from app.core.config import get_settings
from app.db import session as db_session
from app.db.models import Base
from app.ml.inference import get_engine
from app.schemas import HealthOut

FRONTEND_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    get_settings()
    Base.metadata.create_all(bind=db_session.engine)
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Deepfake detection API with Grad-CAM explainability and forensic reports.",
        lifespan=lifespan,
    )

    # Allow Cursor port-forward / cloud preview origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(auth.router, prefix="/api")
    app.include_router(detect.router, prefix="/api")
    app.include_router(history.router, prefix="/api")

    heatmaps = settings.upload_dir / "heatmaps"
    heatmaps.mkdir(parents=True, exist_ok=True)
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/files/heatmaps", StaticFiles(directory=str(heatmaps)), name="heatmaps")
    app.mount("/files/reports", StaticFiles(directory=str(settings.reports_dir)), name="reports")

    @app.get("/api/health", response_model=HealthOut)
    def health() -> HealthOut:
        try:
            eng = get_engine()
            model = eng.model_name
            device = str(eng.device)
            weights_loaded = eng.has_finetuned_weights
            inference_mode = "pytorch" if weights_loaded else "pytorch+forensics"
            status = "ok"
        except Exception:
            model = settings.default_model
            device = settings.device
            weights_loaded = False
            inference_mode = "unavailable"
            status = "degraded"
        return HealthOut(
            status=status,
            app=settings.app_name,
            version=settings.app_version,
            model=model,
            device=device,
            weights_loaded=weights_loaded,
            inference_mode=inference_mode,
        )

    # Serve React build from the same port (works with Cursor port forwarding)
    if FRONTEND_DIST.exists():
        assets = FRONTEND_DIST / "assets"
        if assets.exists():
            app.mount("/assets", StaticFiles(directory=str(assets)), name="frontend-assets")

        @app.get("/{full_path:path}")
        def spa_fallback(full_path: str):
            candidate = FRONTEND_DIST / full_path
            if full_path and candidate.is_file():
                return FileResponse(candidate)
            return FileResponse(FRONTEND_DIST / "index.html")
    else:

        @app.get("/")
        def root() -> dict:
            return {
                "app": settings.app_name,
                "docs": "/docs",
                "health": "/api/health",
                "note": "Frontend build missing. Run: cd frontend && npm run build",
            }

    return app


app = create_app()
