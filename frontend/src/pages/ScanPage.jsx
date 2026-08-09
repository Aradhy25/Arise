import { useState } from "react";
import { Link } from "react-router-dom";
import { api, assetUrl } from "../lib/api";
import { useAuth } from "../lib/auth.jsx";
import DropZone from "../components/DropZone.jsx";
import ResultPanel from "../components/ResultPanel.jsx";

export default function ScanPage() {
  const { isAuthenticated, token } = useAuth();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
  const [preview, setPreview] = useState(null);
  const [modelName, setModelName] = useState("efficientnet");

  async function onUpload(file) {
    setError("");
    setResult(null);
    setLoading(true);
    setPreview(URL.createObjectURL(file));

    const form = new FormData();
    form.append("file", file);
    form.append("model_name", modelName);

    try {
      const path = isAuthenticated ? "/api/detect" : "/api/detect/public";
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
        <div className="animate-rise">
          <h1 className="font-[family-name:var(--font-display)] text-4xl md:text-5xl text-[#0c1f17]">
            Scan any media
          </h1>
          <p className="mt-2 text-[#3d5a4c] max-w-2xl">
            Upload an image, video, or audio clip. No account needed for a free scan.
            Sign in to save history and download forensic PDF reports.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-3 animate-rise-delay">
          <label className="text-sm text-[#3d5a4c]">Visual model</label>
          <select
            value={modelName}
            onChange={(e) => setModelName(e.target.value)}
            className="border border-[#0b3d2e]/15 bg-white px-3 py-2 text-sm"
          >
            <option value="efficientnet">EfficientNet-B0</option>
            <option value="xception">Xception / ResNeXt</option>
            <option value="vit">Vision Transformer</option>
          </select>
          <span className="text-xs text-[#3d5a4c]">Audio files use spectral forensics automatically</span>
        </div>

        <DropZone onFile={onUpload} disabled={loading} />

        {error && (
          <p className="text-sm text-[#b42318] bg-[#b42318]/8 px-3 py-2 border border-[#b42318]/20">
            {error}
          </p>
        )}

        {(loading || result || preview) && (
          <ResultPanel
            loading={loading}
            result={result}
            preview={preview}
            heatmapFallback={result?.heatmap_url ? assetUrl(result.heatmap_url) : null}
          />
        )}
      </main>
    </div>
  );
}
