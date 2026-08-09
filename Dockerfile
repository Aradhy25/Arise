# Production single-container image for DeepGuard AI
FROM node:22-alpine AS frontend
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

COPY backend/app ./app
COPY backend/weights ./weights
RUN mkdir -p uploads/heatmaps reports

# app/main.py resolves FRONTEND_DIST as parents[2]/frontend/dist
# With __file__=/app/app/main.py → /frontend/dist
COPY --from=frontend /frontend/dist /frontend/dist

ENV PYTHONUNBUFFERED=1
ENV DEVICE=cpu
ENV SECRET_KEY=change-me-in-production

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
