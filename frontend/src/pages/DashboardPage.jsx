import { useCallback, useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api, assetUrl } from "../lib/api";
import { useAuth } from "../lib/auth.jsx";
import DropZone from "../components/DropZone.jsx";
import ResultPanel from "../components/ResultPanel.jsx";
import HistoryList from "../components/HistoryList.jsx";

export default function DashboardPage() {
  const { user, token, logout } = useAuth();
  const [models, setModels] = useState([]);
  const [modelName, setModelName] = useState("efficientnet");
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [preview, setPreview] = useState(null);
  const [engineInfo, setEngineInfo] = useState(null);

  const refreshHistory = useCallback(async () => {
    const data = await api("/api/history?limit=12", { token });
    setHistory(data.items || []);
  }, [token]);

  useEffect(() => {
    api("/api/detect/models")
      .then((data) => {
        setModels(data.models || []);
        if (data.default) setModelName(data.default);
        setEngineInfo({
          weights_loaded: data.weights_loaded,
          inference_mode: data.inference_mode,
        });
      })
      .catch(() => {});
    refreshHistory().catch(() => {});
  }, [refreshHistory]);

  async function onUpload(file) {
    setError("");
    setResult(null);
    setLoading(true);
    setPreview(URL.createObjectURL(file));

    const form = new FormData();
    form.append("file", file);
    form.append("model_name", modelName);

    try {
      const data = await api("/api/detect", {
        method: "POST",
        token,
        body: form,
      });
      setResult(data);
      await refreshHistory();
    } catch (err) {
      setError(err.message || "Detection failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen">
      <header className="border-b border-[#0b3d2e]/10 bg-white/50 backdrop-blur-md">
        <div className="mx-auto max-w-6xl px-5 py-4 flex items-center justify-between gap-4">
          <div>
            <p className="font-[family-name:var(--font-display)] text-3xl text-[#0b3d2e] leading-none">
              DeepGuard AI
            </p>
            <p className="text-xs tracking-[0.2em] uppercase text-[#3d5a4c] mt-1">
              Deepfake detection system
            </p>
          </div>
          <div className="flex items-center gap-3 text-sm">
            <Link
              to="/live"
              className="border border-[#0b3d2e]/20 px-3 py-1.5 hover:bg-[#0b3d2e] hover:text-white transition"
            >
              Live camera
            </Link>
            <div className="text-right hidden sm:block">
              <p className="font-medium text-[#0c1f17]">{user?.full_name}</p>
              <p className="text-[#3d5a4c]">{user?.role}</p>
            </div>
            <button
              onClick={logout}
              className="border border-[#0b3d2e]/20 px-3 py-1.5 hover:bg-[#0b3d2e] hover:text-white transition"
            >
              Sign out
            </button>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-6xl px-5 py-8 grid lg:grid-cols-[1.15fr_0.85fr] gap-8">
        <section className="animate-rise space-y-6">
          <div>
            <h1 className="font-[family-name:var(--font-display)] text-4xl md:text-5xl text-[#0c1f17]">
              Analyze media
            </h1>
            <p className="mt-2 text-[#3d5a4c] max-w-xl">
              Upload an image or video. Every file is processed live on the server —
              face crop, PyTorch inference, Grad-CAM heatmap, and a PDF report. No mock results.
            </p>
            {engineInfo && (
              <p className="mt-2 text-xs text-[#1f6b4f]">
                Engine: {engineInfo.inference_mode}
                {engineInfo.weights_loaded ? " · fine-tuned weights loaded" : " · bootstrap / fused mode"}
              </p>
            )}
          </div>

          <div className="flex flex-wrap items-center gap-3">
            <label className="text-sm text-[#3d5a4c]">Model</label>
            <select
              value={modelName}
              onChange={(e) => setModelName(e.target.value)}
              className="border border-[#0b3d2e]/15 bg-white px-3 py-2 text-sm outline-none focus:border-[#1f6b4f]"
            >
              {(models.length
                ? models
                : [
                    { id: "efficientnet", name: "EfficientNet-B0" },
                    { id: "xception", name: "Xception / ResNeXt" },
                    { id: "vit", name: "Vision Transformer" },
                  ]
              ).map((m) => (
                <option key={m.id} value={m.id}>
                  {m.name}
                </option>
              ))}
            </select>
          </div>

          <DropZone onFile={onUpload} disabled={loading} />

          {error && (
            <p className="text-sm text-[#b42318] bg-[#b42318]/8 px-3 py-2 border border-[#b42318]/20">
              {error}
            </p>
          )}

          {(loading || result || preview) && (
            <ResultPanel loading={loading} result={result} preview={preview} />
          )}
        </section>

        <aside className="animate-rise-delay space-y-4">
          <div className="bg-white/65 border border-[#0b3d2e]/10 p-5">
            <h2 className="font-[family-name:var(--font-display)] text-2xl text-[#0b3d2e]">
              Detection history
            </h2>
            <p className="text-sm text-[#3d5a4c] mt-1 mb-4">
              Recent analyses stored in PostgreSQL / SQLite.
            </p>
            <HistoryList
              items={history}
              onSelect={(item) => {
                setResult(item);
                setPreview(assetUrl(item.heatmap_url));
              }}
            />
          </div>

          <div className="bg-[#0b3d2e] text-white p-5">
            <p className="font-[family-name:var(--font-display)] text-2xl">Pipeline</p>
            <ol className="mt-3 space-y-2 text-sm text-white/85">
              <li>1. Media upload</li>
              <li>2. OpenCV frame / image processing</li>
              <li>3. Face detection & crop</li>
              <li>4. PyTorch model inference</li>
              <li>5. Grad-CAM / forensic heatmap</li>
              <li>6. Forensic PDF report</li>
            </ol>
          </div>
        </aside>
      </main>
    </div>
  );
}
