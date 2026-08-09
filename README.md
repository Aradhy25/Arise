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

## Run on your computer

### 1. Clone this branch

```bash
git clone https://github.com/Aradhy25/Arise.git
cd Arise
git checkout cursor/deepguard-ai-c30f
```

### 2. One-port setup (easiest)

**Needs:** Python 3.11+, Node.js 20+, ~5 GB free (PyTorch)

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# PyTorch is large — install it first with a long timeout (fixes Mac timeout errors)
pip install --default-timeout=1000 torch==2.5.1 torchvision==0.20.1
pip install --default-timeout=1000 -r backend/requirements.txt

cd frontend
npm install
npm run build
cd ..

cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

**If `pip install` times out on torch (common on Mac Wi‑Fi):**
```bash
pip install --upgrade pip
pip install --default-timeout=1000 --retries 10 torch==2.5.1 torchvision==0.20.1
pip install --default-timeout=1000 -r backend/requirements.txt
```

**Windows (PowerShell)**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt

cd frontend
npm install
npm run build
cd ..

cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Then open **http://localhost:8000** in your browser.

- Register an account (first user becomes admin)
- Upload an image/video, or use **Live camera**
- API docs: http://localhost:8000/docs

If `backend/weights/efficientnet.pth` is missing:
```bash
python scripts/bootstrap_weights.py
```

### 3. Dev mode (two terminals)

Terminal 1 — API:
```bash
source .venv/bin/activate   # Windows: .\.venv\Scripts\Activate.ps1
cd backend
uvicorn app.main:app --reload --port 8000
```

Terminal 2 — UI with hot reload:
```bash
cd frontend
npm install
npm run dev
```

Open **http://localhost:5173** (Vite proxies `/api` to the backend).

### 4. Docker (optional)

```bash
docker compose up --build
```

- App UI (nginx): http://localhost:3000  
- API: http://localhost:8000  
- Postgres: localhost:5432

## Milestones

1. **M1 — AI prototype** — model architectures, preprocessing, real/fake prediction  
2. **M2 — Video forensics** — frame sampling, face crop, aggregation  
3. **M3 — Explainability + backend** — Grad-CAM, FastAPI, DB, PDF reports  
4. **M4 — Production system** — React dashboard, JWT auth, Docker, tests  

## Inference (real — no mock)

Every upload and live webcam frame runs:

1. OpenCV decode  
2. Face detection + crop  
3. **PyTorch EfficientNet** forward pass  
4. **Grad-CAM** heatmap  
5. Optional forensic auxiliary signals  

Bootstrap weights ship in `backend/weights/efficientnet.pth` (trained by `scripts/bootstrap_weights.py`).  
For research-grade accuracy, retrain on FaceForensics++ / Celeb-DF with `scripts/train.py`.

### Live webcam

Open **Live camera** in the UI (`/live`) or:

```bash
curl -X POST http://localhost:8000/api/detect/live \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@frame.jpg" -F "include_heatmap=true"
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
