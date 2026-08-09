import { useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../lib/api";
import { useAuth } from "../lib/auth.jsx";
import ApiStatus from "../components/ApiStatus.jsx";
import DropZone from "../components/DropZone.jsx";
import ResultPanel from "../components/ResultPanel.jsx";

export default function ScanPage() {
  const { isAuthenticated, token } = useAuth();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
  const [batch, setBatch] = useState(null);
  const [preview, setPreview] = useState(null);
  const [modelName, setModelName] = useState("efficientnet");
  const [ensemble, setEnsemble] = useState(false);
  const [batchMode, setBatchMode] = useState(false);

  async function onUpload(file) {
    setError("");
    setResult(null);
    setBatch(null);
    setLoading(true);
    setPreview(URL.createObjectURL(file));

    const form = new FormData();
    form.append("file", file);
    form.append("model_name", modelName);
    form.append("ensemble", ensemble ? "true" : "false");

    try {
      let path = isAuthenticated ? "/api/detect" : "/api/detect/public";
      if (ensemble && !isAuthenticated) path = "/api/detect/ensemble";
      const data = await api(path, {
        method: "POST",
        token: isAuthenticated ? token : undefined,
        body: form,
      });
      setResult(data);
    } catch (err) {
      setError(err.message || "Detection failed");
    } finally {
      setLoading(false);
    }
  }

  async function onBatch(files) {
    setError("");
    setResult(null);
    setBatch(null);
    setLoading(true);
    setPreview(null);

    const form = new FormData();
    files.slice(0, 8).forEach((f) => form.append("files", f));
    form.append("model_name", modelName);

    try {
      const data = await api("/api/detect/batch", { method: "POST", body: form });
      setBatch(data);
    } catch (err) {
      setError(err.message || "Batch detection failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen">
      <header className="border-b border-[#0b3d2e]/10 bg-white/50 backdrop-blur-md">
        <div className="mx-auto max-w-4xl px-5 py-4 flex items-center justify-between gap-3">
          <Link to="/" className="font-[family-name:var(--font-display)] text-3xl text-[#0b3d2e]">
            DeepGuard AI
          </Link>
          <div className="flex items-center gap-2 text-sm">
            <Link to="/live" className="border border-[#0b3d2e]/20 px-3 py-1.5 hover:bg-[#0b3d2e] hover:text-white transition">
              Live
            </Link>
            {isAuthenticated ? (
              <Link to="/app" className="border border-[#0b3d2e]/20 px-3 py-1.5 hover:bg-[#0b3d2e] hover:text-white transition">
                Dashboard
              </Link>
            ) : (
              <Link to="/auth" className="bg-[#0b3d2e] text-white px-3 py-1.5 font-semibold">
                Sign in
              </Link>
            )}
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-4xl px-5 py-8 space-y-6">
        <ApiStatus />

        <div className="animate-rise">
          <h1 className="font-[family-name:var(--font-display)] text-4xl md:text-5xl text-[#0c1f17]">
            Advanced media scan
          </h1>
          <p className="mt-2 text-[#3d5a4c] max-w-2xl">
            Single or batch upload. Enable ensemble voting across EfficientNet, Xception, and ViT.
            Results include risk tier, Grad-CAM, frame timeline, SHA-256, and JSON export.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-4 animate-rise-delay text-sm">
          <label className="flex items-center gap-2 text-[#3d5a4c]">
            Model
            <select
              value={modelName}
              onChange={(e) => setModelName(e.target.value)}
              disabled={ensemble}
              className="border border-[#0b3d2e]/15 bg-white px-3 py-2"
            >
              <option value="efficientnet">EfficientNet-B0</option>
              <option value="xception">Xception / ResNeXt</option>
              <option value="vit">Vision Transformer</option>
            </select>
          </label>
          <label className="flex items-center gap-2 cursor-pointer">
            <input type="checkbox" checked={ensemble} onChange={(e) => setEnsemble(e.target.checked)} />
            <span>Ensemble (3-model vote)</span>
          </label>
          <label className="flex items-center gap-2 cursor-pointer">
            <input type="checkbox" checked={batchMode} onChange={(e) => setBatchMode(e.target.checked)} />
            <span>Batch mode</span>
          </label>
        </div>

        <DropZone
          disabled={loading}
          multiple={batchMode}
          onFile={onUpload}
          onFiles={onBatch}
        />

        {error && (
          <p className="text-sm text-[#b42318] bg-[#b42318]/8 px-3 py-2 border border-[#b42318]/20">
            {error}
          </p>
        )}

        {(loading || result || preview) && !batch && (
          <ResultPanel loading={loading} result={result} preview={preview} />
        )}

        {batch && (
          <div className="space-y-4 animate-rise">
            <div className="bg-[#0b3d2e] text-white p-4 flex flex-wrap gap-4 text-sm">
              <span>Batch: {batch.total} files</span>
              <span>FAKE: {batch.fake_count}</span>
              <span>REAL: {batch.real_count}</span>
            </div>
            {batch.items.map((item, idx) => (
              <ResultPanel key={idx} loading={false} result={item} preview={null} />
            ))}
          </div>
        )}
      </main>
    </div>
  );
}
