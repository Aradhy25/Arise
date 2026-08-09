# Deploy DeepGuard on Netlify

Netlify = **frontend website only**.  
AI backend (FastAPI + PyTorch) must run on Render / Fly / Railway / VPS.

## Steps

### A. Backend first (Render)

1. https://render.com → New Web Service → this GitHub repo  
2. Docker runtime, Dockerfile `./Dockerfile`  
3. Env: `SECRET_KEY`, `DEVICE=cpu`, `CORS_ORIGINS=*`  
4. Note the URL: `https://xxxx.onrender.com`

### B. Frontend (Netlify)

1. https://app.netlify.com → Add site → Import from Git  
2. Repo: `Aradhy25/Arise`, branch: `cursor/deepguard-ai-c30f`  
3. Build settings come from root `netlify.toml`  
4. Environment variable:

| Key | Value |
|-----|--------|
| `VITE_API_URL` | `https://xxxx.onrender.com` (your Render URL, no trailing slash) |

5. Deploy

Open the Netlify URL → **Scan now** works worldwide.

## Local preview of Netlify build

```bash
cd frontend
VITE_API_URL=http://127.0.0.1:8000 npm run build
npx netlify dev
# or: npx serve dist
```
