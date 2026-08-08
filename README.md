# DeepGuard AI

Forensic deepfake detection system — React + FastAPI + PyTorch + Grad-CAM + PostgreSQL.

## Architecture

```
User (image/video)
        │
        ▼
 React + Vite + Tailwind
        │  REST
        ▼
     FastAPI
   ┌────┼────┐
OpenCV Face  Files
   │    │
   └──┬─┘
      ▼
 PyTorch models
 EfficientNet / Xception / ViT
      │
 Real/Fake + confidence
      │
   Grad-CAM heatmap
      │
 Forensic PDF report
      │
  PostgreSQL / SQLite
```

## Tech stack

| Layer | Choice |
|-------|--------|
| Language | Python 3.12 |
| DL | PyTorch + TorchVision |
| Models | EfficientNet-B0 → Xception/ResNeXt → ViT-B/16 |
| CV | OpenCV |
| Face detection | OpenCV Haar / YuNet / MediaPipe (RetinaFace-ready) |
| XAI | Grad-CAM (+ forensic ELA heatmap fallback) |
| Backend | FastAPI + JWT |
| Frontend | React + Vite + Tailwind CSS |
| DB | PostgreSQL (SQLite for local demo) |
| Reports | ReportLab |
| Deploy | Docker Compose |
| Tests | PyTest |

## Quick start (local)

### Backend

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
cd backend
uvicorn app.main:app --reload --port 8000
```

API docs: http://localhost:8000/docs

### Frontend

```bash
cd frontend
npm install
npm run dev
```

UI: http://localhost:5173

### Docker

```bash
docker compose up --build
```

- Frontend: http://localhost:3000  
- API: http://localhost:8000  
- Postgres: localhost:5432

## Milestones

1. **M1 — AI prototype** — model architectures, preprocessing, real/fake prediction  
2. **M2 — Video forensics** — frame sampling, face crop, aggregation  
3. **M3 — Explainability + backend** — Grad-CAM, FastAPI, DB, PDF reports  
4. **M4 — Production system** — React dashboard, JWT auth, Docker, tests  

## Inference modes

- **With fine-tuned weights** (`backend/weights/<model>.pth`): PyTorch classifier + Grad-CAM  
- **Without weights (default demo)**: forensic heuristics (ELA, frequency, noise, color correlation) so the pipeline stays honest — ImageNet backbones alone are not deepfake detectors  

Train on FaceForensics++ / Celeb-DF:

```bash
python scripts/train.py --model efficientnet --data-dir ./data --epochs 10
python scripts/evaluate.py --model efficientnet --data-dir ./data --weights backend/weights/efficientnet.pth
```

## API overview

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/auth/register` | Create account (first user = admin) |
| POST | `/api/auth/login` | JWT login |
| GET | `/api/auth/me` | Current user |
| POST | `/api/detect` | Upload image/video + model |
| GET | `/api/detect/models` | Available models |
| GET | `/api/history` | Detection history |
| GET | `/api/health` | Health check |

## Datasets (research)

Start with **FaceForensics++** and **Celeb-DF**. Optionally add DFDC / ForgeryNet if storage allows.

## Project layout

```
backend/app/          FastAPI + ML pipeline
frontend/src/         React dashboard
scripts/train.py      Training scaffold
scripts/evaluate.py   Metrics (accuracy, F1, ROC-AUC) — no invented numbers
docker-compose.yml    Full stack
docs/                 Extra notes
```

## Testing

```bash
cd backend
pytest -q
```
