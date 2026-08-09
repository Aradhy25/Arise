import { useCallback, useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../lib/api";
import { useAuth } from "../lib/auth.jsx";

export default function LivePage() {
  const { token, logout, user, isAuthenticated } = useAuth();
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [active, setActive] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
  const [fpsHint, setFpsHint] = useState(1);
  const intervalRef = useRef(null);

  const stop = useCallback(() => {
    setActive(false);
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    const stream = videoRef.current?.srcObject;
    if (stream) {
      stream.getTracks().forEach((t) => t.stop());
      videoRef.current.srcObject = null;
    }
  }, []);

  useEffect(() => () => stop(), [stop]);

  async function start() {
    setError("");
    setResult(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
        audio: false,
      });
      videoRef.current.srcObject = stream;
      await videoRef.current.play();
      setActive(true);

      const tickMs = Math.max(400, Math.round(1000 / fpsHint));
      intervalRef.current = setInterval(() => {
        captureAndDetect();
      }, tickMs);
    } catch (err) {
      setError(err.message || "Camera permission denied");
    }
  }

  async function captureAndDetect() {
    if (busy || !videoRef.current || !canvasRef.current) return;
    const video = videoRef.current;
    if (video.readyState < 2) return;

    const canvas = canvasRef.current;
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.72));
    if (!blob) return;

    setBusy(true);
    try {
      const form = new FormData();
      form.append("file", blob, "frame.jpg");
      form.append("include_heatmap", "true");
      const data = await api("/api/detect/live", {
        method: "POST",
        token: isAuthenticated ? token : undefined,
        body: form,
      });
      setResult(data);
    } catch (err) {
      setError(err.message || "Live detection failed");
    } finally {
      setBusy(false);
    }
  }

  const isFake = result?.prediction === "FAKE";

  return (
    <div className="min-h-screen">
      <header className="border-b border-[#0b3d2e]/10 bg-white/50 backdrop-blur-md">
        <div className="mx-auto max-w-6xl px-5 py-4 flex items-center justify-between gap-4">
          <div>
            <p className="font-[family-name:var(--font-display)] text-3xl text-[#0b3d2e] leading-none">
              DeepGuard AI
            </p>
            <p className="text-xs tracking-[0.2em] uppercase text-[#3d5a4c] mt-1">Live webcam detection</p>
          </div>
          <div className="flex items-center gap-3 text-sm">
            <Link to="/" className="border border-[#0b3d2e]/20 px-3 py-1.5 hover:bg-[#0b3d2e] hover:text-white transition">
              Home
            </Link>
            <Link to="/scan" className="border border-[#0b3d2e]/20 px-3 py-1.5 hover:bg-[#0b3d2e] hover:text-white transition">
              Scan
            </Link>
            {isAuthenticated ? (
              <>
                <span className="hidden sm:inline text-[#3d5a4c]">{user?.full_name}</span>
                <button onClick={logout} className="border border-[#0b3d2e]/20 px-3 py-1.5 hover:bg-[#0b3d2e] hover:text-white transition">
                  Sign out
                </button>
              </>
            ) : (
              <Link to="/auth" className="bg-[#0b3d2e] text-white px-3 py-1.5 font-semibold">
                Sign in
              </Link>
            )}
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-6xl px-5 py-8 grid lg:grid-cols-2 gap-8">
        <section className="animate-rise space-y-4">
          <h1 className="font-[family-name:var(--font-display)] text-4xl text-[#0c1f17]">Real-time scan</h1>
          <p className="text-[#3d5a4c]">
            Your camera frames are sent to the FastAPI backend and scored by the live PyTorch model.
            No canned results — every frame is inferred on the server.
          </p>

          <div className="relative overflow-hidden bg-black aspect-[4/3]">
            <video ref={videoRef} playsInline muted className="h-full w-full object-cover scale-x-[-1]" />
            {!active && (
              <div className="absolute inset-0 flex items-center justify-center text-white/80 text-sm">
                Camera idle
              </div>
            )}
            {busy && <div className="scan-line" />}
          </div>
          <canvas ref={canvasRef} className="hidden" />

          <div className="flex flex-wrap items-center gap-3">
            {!active ? (
              <button onClick={start} className="bg-[#0b3d2e] text-white px-5 py-2.5 font-semibold hover:bg-[#1f6b4f]">
                Start live detection
              </button>
            ) : (
              <button onClick={stop} className="border border-[#b42318] text-[#b42318] px-5 py-2.5 font-semibold hover:bg-[#b42318] hover:text-white">
                Stop
              </button>
            )}
            <label className="text-sm text-[#3d5a4c] flex items-center gap-2">
              Rate
              <select
                value={fpsHint}
                disabled={active}
                onChange={(e) => setFpsHint(Number(e.target.value))}
                className="border border-[#0b3d2e]/15 bg-white px-2 py-1"
              >
                <option value={0.5}>0.5 / sec</option>
                <option value={1}>1 / sec</option>
                <option value={2}>2 / sec</option>
              </select>
            </label>
          </div>
          {error && <p className="text-sm text-[#b42318]">{error}</p>}
        </section>

        <section className="animate-rise-delay space-y-4">
          <div className="bg-white/70 border border-[#0b3d2e]/10 p-6 min-h-[20rem]">
            {!result ? (
              <p className="text-[#3d5a4c]">Waiting for the first live frame…</p>
            ) : (
              <>
                <p className="text-xs tracking-[0.2em] uppercase text-[#3d5a4c]">Live prediction</p>
                <p className={`font-[family-name:var(--font-display)] text-5xl mt-1 ${isFake ? "text-[#b42318]" : "text-[#027a48]"}`}>
                  {result.prediction}
                </p>
                <p className="text-2xl font-semibold mt-2">{(result.confidence * 100).toFixed(1)}% confidence</p>
                <dl className="mt-5 grid grid-cols-2 gap-3 text-sm">
                  <div className="bg-[#f4faf7] px-3 py-2">
                    <dt className="text-[11px] uppercase tracking-wider text-[#3d5a4c]">Mode</dt>
                    <dd className="font-semibold">{result.mode}</dd>
                  </div>
                  <div className="bg-[#f4faf7] px-3 py-2">
                    <dt className="text-[11px] uppercase tracking-wider text-[#3d5a4c]">Latency</dt>
                    <dd className="font-semibold">{result.processing_time_sec}s</dd>
                  </div>
                  <div className="bg-[#f4faf7] px-3 py-2">
                    <dt className="text-[11px] uppercase tracking-wider text-[#3d5a4c]">Model</dt>
                    <dd className="font-semibold">{result.model_name}</dd>
                  </div>
                  <div className="bg-[#f4faf7] px-3 py-2">
                    <dt className="text-[11px] uppercase tracking-wider text-[#3d5a4c]">Version</dt>
                    <dd className="font-semibold">{result.model_version}</dd>
                  </div>
                </dl>
                {result.heatmap_b64 && (
                  <figure className="mt-5">
                    <figcaption className="text-xs uppercase tracking-wider text-[#3d5a4c] mb-2">
                      Grad-CAM heatmap
                    </figcaption>
                    <img
                      src={`data:image/jpeg;base64,${result.heatmap_b64}`}
                      alt="Live Grad-CAM heatmap"
                      className="w-full max-h-72 object-cover"
                    />
                  </figure>
                )}
              </>
            )}
          </div>
        </section>
      </main>
    </div>
  );
}
